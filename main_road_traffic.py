import argparse
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

from script import dataloader, utility, earlystopping
from model import graph_wavenet

parser = argparse.ArgumentParser(description='Graph WaveNet for road traffic prediction')
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')

args = parser.parse_args()
print('Training configs: {}'.format(args))

if args.enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')