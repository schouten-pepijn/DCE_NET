import torch
import torch.utils.data as utils
import processing.processing as processing
import processing.preparation as preparation
import numpy as np
import model
import train
import os
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

def run_analysis(hp):
    
    print(f'device: {hp.device}')

    # create save folders
    processing.create_folder(hp.folder_ids)
    
    # use pretrained network
    if hp.network.pretrained:
        # load network
        print('loading pretrained net')
        net = model.DCE_NET(hp).to(hp.device)
        net.load_state_dict(
            torch.load('pretrained/' + hp.exp_name + '.pt'))
        net.to(hp.device)
        
        # load concentration maps
        print('loading concentration maps')
        dce_test = np.load('pretrained/test_data.npy')
        dce_shape = (65, 20, 16, 256, 256)
        param_shape = dce_shape[1:]

        # load acquisition timing
        print('loading timing')
        hp.acquisition.timing = torch.FloatTensor(
            np.load('pretrained/t_min.npy')).to(hp.device)
        
        # load aif data
        print('loading aif values')
        aif_dicts = np.load('pretrained/aif_dict.npy', allow_pickle=True).item()
        
        # convert AIF model parameters to float values
        aif_vals = processing.aif_loader_pop(aif_dicts,
                                       t0=hp.t0_value,
                                       device=hp.device)
    else:
        # conversion of the DCE signal to concentration signal
        # calculate the DCE timing and AIF parameters
        (timing, 
          dce_train, dce_train_shape,
          dce_test, dce_shape,
          aif_dicts) = preparation.main(hp)
        param_shape = dce_shape[1:]
        
        hp.acquisition.timing = torch.FloatTensor(timing).to(hp.device)
    
        # convert AIF model parameters to float values
        aif_vals = processing.aif_loader_pop(aif_dicts,
                                       t0=hp.t0_value,
                                       device=hp.device)
    
        # mask extreme concentration signals in training data
        if hp.mask_extremes:
            dce_train = processing.mask_extremes(dce_train,
                                           hp.mask_val)
    
        # mask minimal enhancing voxels in training data
        if hp.mask_zeros:
            dce_train, _ = processing.mask_zeros(dce_train,
                                           hp.mask_cutoff)
    
        # train the network
        net = train.train(dce_train, hp, aif_vals)
        
        # save the network parameters
        torch.save(net.state_dict(),
                   os.path.join(hp.folder_ids[0], f'{hp.exp_name}.pt'))

    # make prediction on full data set
    out = predict_DCE(dce_test, net, hp, aif_vals)

    # export the parameter maps
    param_maps = processing.param_map(out,
                                      (dce_shape, param_shape),
                                      save_path=hp.folder_ids[1],
                                      save_nii=True)

    return param_maps


"""
this function performs the interference on the main data set
"""
def predict_DCE(C1, net, hp, aif_vals, one_dim=True):
    print('performing interference on test set')
    net.eval()

    C1[np.isnan(C1)] = 0

    # load the data in batches
    inferloader = utils.DataLoader(C1,
                                   batch_size=hp.training.val_batch_size,
                                   shuffle=False,
                                   drop_last=False)

    # perform inference on the trained network
    ke = torch.zeros((len(C1)))
    ve = torch.zeros((len(C1)))
    X_pred = torch.zeros((*C1.shape,))
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.float()
            X_batch = X_batch.to(hp.device)
            X_dw, ket, dtt, vet, vpt = net(X_batch, aif_vals)

            i_1 = i * len(X_batch)
            i_2 = (i + 1) * len(X_batch)

            ke[i_1:i_2] = ket.cpu().squeeze()
            ve[i_1:i_2] = vet.cpu().squeeze()
            X_pred[i_1:i_2, :] = X_dw.cpu().squeeze()

    X_pred = X_pred.numpy()
    ktrans = ke.numpy() * ve.numpy()
    params = [X_pred, ktrans]

    return params