import numpy as np

a = np.random.rand(1024,2259,20)
b = np.random.rand(1024,2259)
newAns = np.zeros(a.shape)
for i in range(a.shape[2]):
    newAns[:,:,i] = a[:, :, i] * b

ans = a * b.unsqueeze(2)
print("adsfadsf")
