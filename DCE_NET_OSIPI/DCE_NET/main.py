"""
November 2021 by Pepijn Schouten & Oliver Gurney-Champion
pschouten@amsterdamumc.nl & o.j.gurney-champion@amsterdamumc.nl
Solved the ktrans parameter maps for the OSIPI DCE CHALLENGE DATA.
Analysed concentration curves with a recurrent neural network using gru's.
Copyright (C) 2021 by Oliver Gurney-Champion and Matthew Orton
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
dependencies:
numpy
pytorch
joblib
scipy
matplotlib
nibabel
tqdm
pydicom
"""
import hyperparams
import numpy as np
import torch
import argparse
import analysis as ana

# fixed seed point to eliminate stochastic variations
np.random.seed(42)
torch.manual_seed(42)

# parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--njobs', type=int, default=10)
parser.add_argument('--cpu', action='store_false', default=True)
parser.add_argument('--pretrained', action='store_true', default=False)
args = parser.parse_args()

# setting parser arguments
hp = hyperparams.Hyperparams()
hp.network.pretrained = args.pretrained

if hp.network.pretrained:
    hp.device = torch.device("cpu")
    # hp.device = torch.device("cuda:0")
    print(f'network type: {hp.network.nn}')
else:
    hp.n_jobs = args.njobs
    hp.use_cuda = args.cpu
    hp.device = torch.device("cuda:0" if hp.use_cuda else "cpu")
    print(f'pretrained: {hp.network.pretrained}')
    print(f'epochs: {hp.training.epochs}')
    print(f'iterations per epoch: {hp.training.totalit}')
    print(f'training batch size: {hp.training.batch_size}')

# run the analysis module
ana.run_analysis(hp)
