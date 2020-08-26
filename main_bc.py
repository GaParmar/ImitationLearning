import os, sys, time, pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CarDataset
from networks import LinearPolicy

"""
Configuration parameters
"""
train_bs           = 16
num_workers        = 8
roots              = [ "/home/gparmar/Desktop/github_gaparmar/CarSimulator/output/expert_0",
                        "/home/gparmar/Desktop/github_gaparmar/CarSimulator/output/expert_1",
                        "/home/gparmar/Desktop/github_gaparmar/CarSimulator/output/expert_2",
                        "/home/gparmar/Desktop/github_gaparmar/CarSimulator/output/expert_3",
                        "/home/gparmar/Desktop/github_gaparmar/CarSimulator/output/expert_4",
                     ]
train_epochs       = 500
train_lr           = 0.0002




ds_train = CarDataset(roots, split="train")
dl_train = DataLoader(ds_train, batch_size=train_bs,
                        shuffle=True, pin_memory=True, 
                        num_workers=num_workers)
ds_test = CarDataset(roots, split="test")
dl_test = DataLoader(ds_test, batch_size=train_bs,
                        shuffle=True, pin_memory=True, 
                        num_workers=num_workers)

policy = LinearPolicy(output_ch=2).cuda()
opt = torch.optim.Adam(policy.parameters(), lr=train_lr,
                    weight_decay=1e-5)

device = torch.device("cuda")

L_train, L_test = [], []

for epoch in range(train_epochs):
    policy = policy.train()
    train_loss = 0.0
    for idx, batch in enumerate(dl_train, 1):
        # Make all the gradients zero
        opt.zero_grad()
        img = batch["image"].cuda()
        # predict throttle / steer using current weights
        pred_throttle, pred_steer = policy(img)
        # Computer Mean Squared Difference
        mse_loss = F.mse_loss(pred_throttle.view(-1), batch["throttle"].to(device))
        mse_loss += F.mse_loss(pred_steer.view(-1), batch["steer"].to(device))
        # Compute the gradients of the MSE w.r.t the model weights
        mse_loss.backward()
        # upate model weights using the gradient
        opt.step()
        train_loss += mse_loss.item()
    train_loss = train_loss/len(ds_train)
    L_train.append(train_loss)

    test_loss = 0.0
    policy = policy.eval()
    for idx, batch in enumerate(dl_test, 1):
        with torch.no_grad():
            img = batch["image"].cuda()
            pred_throttle, pred_steer = policy(img)
            mse_loss = F.mse_loss(pred_throttle.view(-1), batch["throttle"].to(device))
            mse_loss += F.mse_loss(pred_steer.view(-1), batch["steer"].to(device))
            test_loss += mse_loss.item()
        # pbar.set_description(f"epoch:{epoch:3d}\tit:{idx:4d}\ttest_loss:{mse_loss.item():.2f}\t\t")
    test_loss = test_loss/len(ds_test)
    L_test.append(test_loss)

    print(f"{epoch}:: total train_loss: {train_loss}\ttest_loss: {test_loss}")
    if epoch%50 == 0:
        # save model 
        torch.save(policy.state_dict(), f"model_{epoch}.sd")


# plot the losses
plt.plot(L_train, label="train loss")
plt.plot(L_test, label="test loss")
plt.legend()
save_path = os.path.join(f"losses.png")
plt.savefig(save_path) 
