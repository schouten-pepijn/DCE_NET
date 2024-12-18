import hyperparams
import argparse
import torch
import numpy as np
import simulations as sim

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[160, 160, 160])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--supervised', action='store_true', default=False)
parser.add_argument('--results', action='store_true', default=False)

args = parser.parse_args()

hp = hyperparams.Hyperparams()

# hp.training.lr = args.lr
# hp.training.batch_size = args.batch_size
# hp.network.nn = args.nn
# hp.network.layers = args.layers
# hp.network.attention = args.attention
# hp.network.bidirectional = args.bidirectional
# hp.supervised = args.supervised

# create save name for framework
hp.exp_name = ''
arg_dict = vars(args)
for i, arg in enumerate(arg_dict):
    if i == len(arg_dict)-2:
        hp.exp_name += str(arg_dict[arg])
        break
    else:
        hp.exp_name += '{}_'.format(arg_dict[arg])

print(hp.exp_name)
print('network type; ', hp.network.nn)


sim.run_simulations(hp, SNR='all')
