import flwr as fl
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Train on 25 clients (each round)
    fraction_eval=0.5,  # Evaluate on 50 clients (each round)
    min_fit_clients=2,
    min_eval_clients=1,
    min_available_clients=3
)
if __name__ == "__main__":
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=strategy)