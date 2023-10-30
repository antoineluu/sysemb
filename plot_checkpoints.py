import torch
import os

# Python program to explain os.mkdir() method  
    
# importing os module  
import os 
import matplotlib.pyplot as plt
import torch
fig = plt.figure(figsize=(25,25))

directory = "checkpoints"
parent_dir = "./"
path = os.path.join(parent_dir, directory)
dirlist = os.listdir(path)
for string in dirlist:
    file = os.path.join(path,string)
    cp = torch.load(file)
    # print(cp["total_trainable_params"],cp["accuracy"])
    plt.title("Performance comparison of trainings")
    plt.scatter(cp["total_trainable_params"], list(cp["accuracy"])[0], label=str(string))
    plt.legend()
plt.show()


    