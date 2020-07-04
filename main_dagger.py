import os, sys, time, pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.data_collector import *
from utils.dataset import CarDataset
from utils.networks import LinearPolicy

"""
Configuration parameters
"""
sim_path      = ".tmp/sim/DonkeySimLinux/donkey_sim.x86_64"
base_out_path = "output/dagger/"
batch_size    = 32
num_workers   = 8
train_epochs  = 100
train_lr      = 0.0003

dc = DataCollector(sim_path, controller="xbox", angle_scale=0.3, throttle_scale=0.4)
d_ds, d_losses = {}, {}


policy = LinearPolicy(output_ch=2).cuda()
opt = torch.optim.Adam(policy.parameters(), lr=train_lr,
                    weight_decay=1e-5)


# 1. Collect expert training policy if it does not exist
curr_out_folder = os.path.join(base_out_path, "trajectory_0")
if not os.path.exists(curr_out_folder):
    os.makedirs(curr_out_folder)
dc.collect_data(output_folder=curr_out_folder, override=True, buffer_length=100, duration=60, refresh_rate=10)
dc.reset()

# 2. train on trajectory 0
d_ds[0] = CarDataset([os.path.join(base_out_path, "trajectory_0")], split="both")
dl = DataLoader(d_ds[0], batch_size=batch_size,
                shuffle=True, pin_memory=True, 
                num_workers=num_workers)
d_losses[0] = []
pbar = tqdm(range(train_epochs))
for epoch in pbar:
    policy = policy.train()
    train_loss = 0.0
    for idx, batch in enumerate(dl):
        opt.zero_grad()
        img = batch["image"].cuda()
        pred_throttle, pred_steer = policy(img)
        mse_loss = F.mse_loss(pred_throttle.view(-1), batch["throttle"].cuda())
        mse_loss += F.mse_loss(pred_steer.view(-1), batch["steer"].cuda())
        mse_loss.backward()
        opt.step()
        train_loss += mse_loss.item()
    pbar.set_description(f"epoch:{epoch}  loss:{train_loss:.3f}")
    d_losses[0].append(train_loss/len(d_ds[0]))
plot_file = os.path.join(base_out_path, "losses", "trajectory_0.png")
if not os.path.exists(os.path.join(base_out_path, "losses")):
    os.makedirs(os.path.join(base_out_path, "losses"))
plt.plot(d_losses[0], label="trajectory 0 losses")
plt.legend()
plt.savefig(plot_file) 

# 3. Collect expert data with current policy rollout
policy.train()
dc.set_policy(policy)
curr_out_folder = os.path.join(base_out_path, "trajectory_1")
if not os.path.exists(curr_out_folder):
    os.makedirs(curr_out_folder)
dc.collect_data(output_folder=curr_out_folder, override=True, 
                buffer_length=100, duration=60, refresh_rate=10,
                use_policy=True)
dc.reset()