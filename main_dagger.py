import os, sys, time, pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# import torch 
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from dataset import CarDataset
# from networks import LinearPolicy
import gym
import gym_donkeycar

sim_path      = "../CarSimulator/sim/DonkeySimLinux/donkey_sim.x86_64"
env_name      = "donkey-mountain-track-v0"
host          = "127.0.0.1"
port          = 9092
env  = gym.make(env_name, exe_path=sim_path,
                host=host, port=port)


# 1. Collect expert training policy