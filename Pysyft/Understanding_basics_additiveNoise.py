import torch as torch
import syft as sy
hook = sy.TorchHook(torch)
# jake = sy.VirtualWorker(hook, id="jake")
# print("Jake has: " + str(jake._objects))



# x = torch.tensor([1, 2, 3, 4, 5])
# x = x.send(jake)
# print("x: " + str(x))
# print("Jake has: " + str(jake._objects))


# x = x.get()
# print("x: " + str(x))
# print("Jake has: " + str(jake._objects))



# john = sy.VirtualWorker(hook, id="john")
# x = x.send(jake)
# x = x.send(john)
# print("x: " + str(x))
# print("John has: " + str(john._objects))
# print("Jake has: " + str(jake._objects))



# import random

# # setting Q to a very large prime number
# Q = 23740629843760239486723


# def encrypt(x, n_share=3):
#     r"""Returns a tuple containg n_share number of shares
#     obtained after encrypting the value x."""

#     shares = list()
#     for i in range(n_share - 1):
#         shares.append(random.randint(0, Q))
#     shares.append(Q - (sum(shares) % Q) + x)
#     return tuple(shares)

# def decrypt(shares):
#     r"""Returns a value obtained by decrypting the shares."""

#     return sum(shares) % Q

# print("Shares: " + str(encrypt(30)))
# print("Decrypted: " + str(decrypt(encrypt(30))))




jake = sy.VirtualWorker(hook, id="jake")
john = sy.VirtualWorker(hook, id="john")
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

jake.add_workers([john, secure_worker])
john.add_workers([jake, secure_worker])
secure_worker.add_workers([jake, john])

print("Jake has: " + str(jake._objects))
print("John has: " + str(john._objects))
print("Secure_worker has: " + str(secure_worker._objects))