from collections import OrderedDict
from typing import Dict, List, Tuple
import libs.parameters as parameters
import os
import numpy as np
import torch
import libs
from tools import surrogate_mapping
import json
import sys
import flwr as fl
import main
import shutil
import tools.evaluate as evalaute_one
import tools.evaluate_ensemble as evaluate_ens

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_surrogate_mapping(data_dir, g_config, seed):
    dataset = g_config['dataset']
    surrogate_threshold = g_config['surrogate_threshold']
    arch = g_config['arch']
    tmp_model_dir = os.path.join(
        data_dir, dataset, f'deepxml.{arch}', f"{surrogate_threshold}.{seed}")
    data_dir = os.path.join(data_dir, dataset)
    try:
        os.makedirs(tmp_model_dir, exist_ok=False)
        surrogate_mapping.run(
            feat_fname=os.path.join(data_dir, g_config["trn_feat_fname"]),
            lbl_fname=os.path.join(data_dir, g_config["trn_label_fname"]),
            feature_type=g_config["feature_type"],
            method=g_config['surrogate_method'],
            threshold=g_config['surrogate_threshold'],
            seed=seed,
            tmp_dir=tmp_model_dir)
    except FileExistsError:
        print("Using existing data for surrogate task!")
    finally:
        data_stats = json.load(
            open(os.path.join(tmp_model_dir, "data_stats.json")))
        mapping = os.path.join(
            tmp_model_dir, 'surrogate_mapping.txt')
    return data_stats, mapping

def evaluate(g_config, data_dir, pred_fname, filter_fname=None, betas=-1, n_learners=1):
    if n_learners == 1:
        func = evalaute_one.main
    else:
        func = evaluate_ens.main

    dataset = g_config['dataset']
    data_dir = os.path.join(data_dir, dataset)
    A = g_config['A']
    B = g_config['B']
    if 'save_top_k' in g_config:
        top_k = g_config['save_top_k']
    else:
        top_k = g_config['top_k']
    ans = func(
        tst_label_fname=os.path.join(
            data_dir, g_config["tst_label_fname"]),
        trn_label_fname=os.path.join(
            data_dir, g_config["trn_label_fname"]),
        pred_fname=pred_fname,
        A=A, 
        B=B,
        top_k=top_k,
        filter_fname=filter_fname, 
        betas=betas, 
        save=g_config["save_predictions"])
    return ans

class DeepXMLClient(fl.client.NumPyClient):
    """Flower client implementing Extreme classification using
    PyTorch."""
    def __init__(
        self,
        model_type,
        work_dir,
        version,
        config,
        seed,
        innerDirectory
    ) -> None:
        self.model_type = model_type
        self.work_dir = work_dir
        
        self.config = json.load(open(config))
        if "," in seed:
            self.seed = list(map(int, seed.split(",")))
            self.version = version
        else:
            self.seed = int(seed)
            self.version = f"{version}_{seed}"
        self.innerDirectory = innerDirectory
        self.data_dir = os.path.join(self.work_dir, 'data')
        
        self.args = self.GetParamsObject()
        self.args.mode = 'CreateModel'
        self.model = main.main(self.args)
        dataset = self.config['global']['dataset']
        self.filter_fname = os.path.join(self.data_dir, dataset, 'filter_labels_test.txt')
    
    def GetParamsObject(self):
        g_config = self.config['global']
        dataset = g_config['dataset']
        arch = g_config['arch']
        use_reranker = g_config['use_reranker']
        self.result_dir = os.path.join(
            self.work_dir, 'results', 'DeepXML', arch, dataset, f'v_{self.version}')
        self.model_dir = os.path.join(
            self.work_dir, 'models', 'DeepXML', arch, dataset, f'v_{self.version}')
        _args = parameters.Parameters("Parameters")
        _args.parse_args()
        _args.update(self.config['global'])
        _args.update(self.config['surrogate'])
        _args.params.seed = self.seed
        args = _args.params
        args.data_dir = self.data_dir
        args.model_dir = os.path.join(self.model_dir, 'surrogate')
        args.result_dir = os.path.join(self.result_dir, 'surrogate')
        data_stats, args.surrogate_mapping = create_surrogate_mapping(self.data_dir, g_config, self.seed)
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        args.arch = os.path.join(self.innerDirectory, f'run_scripts/{arch}.json')
        temp = data_stats['surrogate'].split(",")
        args.num_labels = int(temp[2])
        args.vocabulary_dims = int(temp[0])
        self.data_stats = data_stats
        self.arch = args.arch
        return args

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.net.state_dict().items()]

    def set_parameters(self, parameters_model: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.net.state_dict().keys(), parameters_model)})
        
        self.model.net.load_state_dict(state_dict, strict=True)
        g_config = self.config['global']
        dataset = g_config['dataset']
        arch = g_config['arch']
        _args = parameters.Parameters("Parameters")
        _args.parse_args()
        _args.params.seed = self.seed
        _args.update(self.config['extreme'])
        args = _args.params
        args.surrogate_mapping = None
        result_dir = os.path.join(
        self.work_dir, 'results', 'DeepXML', arch, dataset, f'v_{self.version}')
        model_dir = os.path.join(
            self.work_dir, 'models', 'DeepXML', arch, dataset, f'v_{self.version}')
        params = args
        data_dir = self.data_dir
        data={'X': None, 'Y': None}
        learning_rate=params.learning_rate
        num_epochs=params.num_epochs
        trn_feat_fname=params.trn_feat_fname
        val_feat_fname=params.val_feat_fname
        trn_label_fname=params.trn_label_fname
        val_label_fname=params.val_label_fname
        batch_size=params.batch_size
        feature_type=params.feature_type
        num_workers=params.num_workers
        normalize_features=params.normalize
        normalize_labels=params.nbn_rel
        shuffle=params.shuffle
        validate=params.validate
        beta=params.beta
        init_epoch=params.last_epoch
        keep_invalid=params.keep_invalid
        shortlist_method=params.shortlist_method
        validate_after=params.validate_after
        feature_indices=params.feature_indices
        use_intermediate_for_shorty=params.use_intermediate_for_shorty
        label_indices=params.label_indices
        trn_pretrained_shortlist = None
        val_pretrained_shortlist = None
        trn_pretrained_shortlist=trn_pretrained_shortlist
        val_pretrained_shortlist=val_pretrained_shortlist
        surrogate_mapping=params.surrogate_mapping
        args.model_dir = os.path.join(model_dir, 'extreme')
        args.result_dir = os.path.join(result_dir, 'extreme')
        self.model.model_dir = args.model_dir
        self.model.shortlist_size = args.num_nbrs
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        self.model._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=trn_feat_fname,
            fname_labels=trn_label_fname,
            data=data,
            mode='train',
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            size_shortlist=self.model.shortlist_size,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            shortlist_method=shortlist_method,
            feature_type=feature_type,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping,
            pretrained_shortlist=trn_pretrained_shortlist,
            _type='shortlist'
        )


        self.model.save(args.model_dir, 'model')

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        # train intermediate representation
        train_time = 0
        model_size = 0
        avg_prediction_time = 0
        _args = libs.parameters.Parameters("Parameters")
        config = self.config
        _args.parse_args()
        _args.update(config['global'])
        _args.update(config['surrogate'])
        _args.params.seed = self.seed
        args = self.args
        args.mode = 'train'
        args.arch = self.arch
        temp = self.data_stats['surrogate'].split(",")
        args.num_labels = int(temp[2])
        args.vocabulary_dims = int(temp[0])
        #_train_time, model_size, _modelTrained_1 = main.main(args)
        #train_time += _train_time
        _args.update(config['extreme'])
        args = _args.params
        args.surrogate_mapping = None
        args.model_dir = os.path.join(self.model_dir, 'extreme')
        args.result_dir = os.path.join(self.result_dir, 'extreme')
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

        args.mode = 'train'
        args.arch = self.arch
        temp = self.data_stats['extreme'].split(",")
        args.num_labels = int(temp[2])
        args.vocabulary_dims = int(temp[0])
        args.data_dir = self.data_dir
        
        _train_time, _model_size, _modeltrained_2 = main.main(args)
        train_time += _train_time
        model_size += _model_size



        #self.set_parameters(parameters)

        return self.get_parameters(), 2000, {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        #loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        #return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
        _args = libs.parameters.Parameters("Parameters")
        config = self.config
        _args.parse_args()
        _args.update(config['global'])
        _args.update(config['surrogate'])
        _args.params.seed = self.seed
        args = self.args
        args.mode = 'train'
        args.arch = self.arch
        temp = self.data_stats['surrogate'].split(",")
        args.num_labels = int(temp[2])
        args.vocabulary_dims = int(temp[0])
        #_train_time, model_size, _modelTrained_1 = main.main(args)
        #train_time += _train_time
        _args.update(config['extreme'])
        args = _args.params
        args.surrogate_mapping = None
        args.model_dir = os.path.join(self.model_dir, 'extreme')
        args.result_dir = os.path.join(self.result_dir, 'extreme')
        args.pred_fname = 'tst_predictions'
        args.mode = 'predict'
        args.data_dir = self.data_dir
        predicted_labels, prediction_time, _avg_pred_time, stats = main.main(args)
        maxAccuracy = 0
        for k,v in stats.items():
            maxAccuracy = max(maxAccuracy, max(v[0]))
        return (float(0),predicted_labels, stats)



def main_function() -> None:
    """Get model parameters and run One training session"""
    model_type = sys.argv[1]
    work_dir = sys.argv[2]
    version = sys.argv[3]
    config = f"{sys.argv[4]}/{model_type}/{version}.json"
    seed = sys.argv[5]
    innerDirectory = sys.argv[6]
    # Start client
    client = DeepXMLClient(model_type, work_dir, version, config, seed, innerDirectory)
    client.set_parameters(client.get_parameters())
    #client.fit(client.get_parameters(), {})
    client.evaluate(client.get_parameters(), {})
    fl.client.start_numpy_client("0.0.0.0:8080", client)

if __name__ == '__main__':
    main_function()