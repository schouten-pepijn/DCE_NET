import torch
import torch.utils.data as utils
import numpy as np
import pickle
import train
import os
import nibabel as nib
from tqdm import tqdm


def run_simulations(hp, SNR=15, eval=False, var_seq=False):
    print(hp.device)
   
    hp.acquisition.timing = torch.FloatTensor(np.load(
       '/home/pschouten/scratch/arrays/timing/t_min.npy')).to(hp.device)
       
    dce_signal_noisy = np.concatenate((
       np.load(
           '/scratch/pschouten/arrays/c_map/cmap_gth_True_trimmed_True_enh_True_clin.npy'),
       np.load(
           '/scratch/pschouten/arrays/c_map/cmap_gth_True_trimmed_True_enh_True_synth.npy')), axis=1)
       
    dce_shape = dce_signal_noisy.shape
    param_shape = dce_shape[1:]
    dce_signal_noisy = dce_signal_noisy.reshape(65, -1).T

    if hp.full_interference:
        dce_signal_test = np.concatenate((
           np.load(
               '/scratch/pschouten/arrays/c_map/cmap_gth_True_trimmed_False_enh_False_clin.npy'),
           np.load(
               '/scratch/pschouten/arrays/c_map/cmap_gth_True_trimmed_False_enh_False_synth.npy')), axis=1)
        dce_shape_test = dce_signal_test.shape
        param_shape_test = dce_shape_test[1:]
        dce_signal_test = dce_signal_test.reshape(65, -1).T
    
    if hp.mask_extremes:
        low, high = hp.mask_vals[0], hp.mask_vals[1]
        print(f'masking extreme signal voxels < {low} and > {high}')
        mask_low = (dce_signal_noisy < high).all(axis=1)
        mask_high = (dce_signal_noisy > low).all(axis=1)
        mask = np.logical_and(mask_low, mask_high)
        dce_signal_noisy = np.where(mask[:, np.newaxis], dce_signal_noisy, 0)
    
    if hp.mask_zeros:
        if hp.mask_cutoff == 0:
            print('masking zero signal voxels')
            mask = (dce_signal_noisy != 0).any(axis=1)
            dce_signal_noisy = dce_signal_noisy[mask, :]
        else:
            print(f'masking mean < {hp.mask_cutoff} signal voxels')
            mask = np.mean(dce_signal_noisy, axis=1) > hp.mask_cutoff
            dce_signal_noisy = dce_signal_noisy[mask, :]
        print(f'data size out: {dce_signal_noisy.shape}')
    else:
        mask = None
    
    if hp.aif_type == 'pop':
        with open('/scratch/pschouten/arrays/aif/cos4_pop/aif_pop_cos4.pkl',
                  'rb') as file:
            aif_dicts = pickle.load(file)
        aif_vals = torch.FloatTensor([0 if key == 't0'
                                      else aif_dicts[key]
                                      for key
                                      in ('t0', 'ab', 'mb', 'ae', 'me')])
        aif_vals = aif_vals.to(hp.device)

    net = train.train(dce_signal_noisy, hp, aif_vals)

    if not os.path.isdir('pretrained'):
        os.makedirs('pretrained')
    torch.save(net.state_dict(), 'pretrained/pretrained_'+hp.exp_name+'.pt')

    if hp.full_interference:
        if hp.aif_type == 'pop':
            out = predict_DCE(
                dce_signal_test, net, hp, aif_vals)
    else:
        if hp.aif_type == 'pop':
            out = predict_DCE(
                dce_signal_noisy, net, hp, aif_vals)

    if hp.full_interference:
        param_maps = param_map(out,
                                (dce_shape_test, param_shape_test),
                                mask=None,
                                save_path='/scratch/pschouten/DCE-NET/' \
                                          'pop/DCE-NET-1/results/',
                                save_nii=True,
                                save_np=True)
    else:
        if hp.mask_zeros:
            param_maps = param_map(out, 
                                        (dce_shape, param_shape),
                                        mask=mask,
                                        save_path='/scratch/pschouten/DCE-NET/' \
                                                  'pop/DCE-NET-1/results/',
                                        save_nii=True,
                                        save_np=True)
        else:
            param_maps = param_map(out,
                                        (dce_shape, param_shape),
                                        mask=None,
                                        save_path='/scratch/pschouten/DCE-NET/' \
                                                  'pop/DCE-NET-1/results/',
                                        save_nii=True,
                                        save_np=True)
    return param_maps
    
    # return param_results
    return None


def predict_DCE(C1, net, hp, aif_vals):
    net.eval()

    C1[np.isnan(C1)] = 0

    inferloader = utils.DataLoader(C1,
                                   batch_size=hp.training.val_batch_size,
                                   shuffle=False,
                                   drop_last=False)

    # perform inference
    ke, dt, ve, vp, ktrans = (torch.zeros((len(C1))) for _ in range(5))
    X = torch.zeros((*C1.shape,))
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.float()
            X_batch = X_batch.to(hp.device)
            
            if hp.aif_type == 'pop':
                X_dw, ket, dtt, vet, vpt = net(X_batch, aif_vals)

            i_1 = i * len(X_batch)
            i_2 = (i + 1) * len(X_batch)

            ke[i_1:i_2] = ket.cpu().squeeze()
            dt[i_1:i_2] = dtt.cpu().squeeze()
            ve[i_1:i_2] = vet.cpu().squeeze()
            vp[i_1:i_2] = vpt.cpu().squeeze()
            X[i_1:i_2, :] = X_dw.cpu().squeeze()

    ke = ke.numpy()
    dt = dt.numpy()
    ve = ve.numpy()
    vp = vp.numpy()
    X = X.numpy()
    ktrans = ke * ve
    
    params = [X, ke, dt, ve, vp, ktrans]

    return params


def sim_results(paramsNN_full, hp, kep, ve, vp, Tonset, Hct=None):
    # calculate the random and systematic error of every parameter
    rep1 = hp.acquisition.rep1 - 1
    error_ke = paramsNN_full[0] - np.squeeze(kep)
    randerror_ke = np.std(error_ke)
    syserror_ke = np.mean(error_ke)
    del error_ke

    error_ve = paramsNN_full[2] - np.squeeze(ve)
    randerror_ve = np.std(error_ve)
    syserror_ve = np.mean(error_ve)
    del error_ve

    error_vp = paramsNN_full[3] - np.squeeze(vp)
    randerror_vp = np.std(error_vp)
    syserror_vp = np.mean(error_vp)
    del error_vp

    error_dt = paramsNN_full[1] - (np.squeeze(Tonset) + rep1 * hp.simulations.time) / 60
    randerror_dt = np.std(error_dt)
    syserror_dt = np.mean(error_dt)
    del error_dt, paramsNN_full

    normke = np.mean(kep)
    normve = np.mean(ve)
    normvp = np.mean(vp)
    normdt = np.mean(Tonset / 60)
    print('ke_sim, dke_lsq, sys_ke_lsq, dke, sys_ke')
    print([normke, '  ', randerror_ke, '  ', syserror_ke])
    print([normve, '  ', randerror_ve, '  ', syserror_ve])
    print([normvp, '  ', randerror_vp, '  ', syserror_vp])
    print([normdt, '  ', randerror_dt, '  ', syserror_dt])

    return np.array([[randerror_ke, syserror_ke],
                     [randerror_ve, syserror_ve],
                     [randerror_vp, syserror_vp],
                     [randerror_dt, syserror_dt]])


def param_map(params, shapes, mask=None,
              save_path=None, save_np=False, save_nii=True):
    X_pred, ke, dt, ve, vp, ktrans = params
    dce_shape, param_shape = shapes
    
    if mask is not None:
        ke_map, dt_map, ve_map, vp_map, ktrans_map = (
            np.zeros((np.prod(param_shape))) for _ in range(5))
        X_pred_map = np.zeros((len(mask), X_pred.shape[-1]))

        ke_map[mask], ke_map[~mask] = (ke, 0)
        dt_map[mask], dt_map[~mask] =(dt, 0)
        ve_map[mask], ve_map[~mask] = (ve, 0)
        vp_map[mask], vp_map[~mask] = (vp, 0)
        ktrans_map[mask], ktrans_map[~mask] = (ktrans, 0) 
        X_pred_map[mask, :], X_pred_map[~mask, :] = (X_pred, 0)

        ke_map = ke_map.reshape(*param_shape)
        dt_map = dt_map.reshape(*param_shape)
        ve_map = ve_map.reshape(*param_shape)
        vp_map = vp_map.reshape(*param_shape)
        ktrans_map = ktrans_map.reshape(*param_shape)
        X_pred_map = np.swapaxes(X_pred_map.T.reshape(*dce_shape), 0 ,1)
    else:
        ke_map = ke.reshape(*param_shape)
        dt_map = dt.reshape(*param_shape)
        ve_map = ve.reshape(*param_shape)
        vp_map = vp.reshape(*param_shape)
        ktrans_map = ktrans.reshape(*param_shape)
        X_pred_map = np.swapaxes(X_pred.T.reshape(*dce_shape), 0, 1)
    params = [X_pred_map, ke_map, dt_map, ve_map, vp_map, ktrans_map]

    if save_path is not None:  
        if save_np:
            for tag, item in zip(
                    ('C_pred', 'ke', 'dt', 've', 'vp', 'ktrans'), params):
                print(f'saving {tag} map to numpy')
                np.save(os.path.join(save_path, f'{tag}.npy'), item)
    
        if save_nii:
            for tag, item in zip(
                    ('ke_map', 'dt_map', 've_map', 'vp_map', 'ktrans_map'), params[1:]):
                array_to_nii(item,
                             path=save_path,
                             file_name=f'{tag}.nii.gz',
                             transpose=True)

            for i, X in enumerate(X_pred_map):
                array_to_nii(X,
                             path=save_path,
                             file_name=f'C_pred_pat_{i//2+1}_vis_{i%2+1}.nii.gz',
                             transpose=True)
    return params


def array_to_nii(data, path, file_name, transform=None, transpose=False):
        # x y z t
        print(f'saving {file_name} to nifti')
        if isinstance(transform, tuple):
            print(f'in shape: {data.shape}')
            data = np.moveaxis(data, transform[0], transform[1])
            print(f'out shape: {data.shape}')
        if transpose:
            data = data.T

        if not '.nii.gz' in file_name:
            if '.nii' in file_name:
                file_name.replace('.nii', '.nii.gz')
            else:
                file_name += '.nii.gz'
        ni_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(ni_img, os.path.join(path, file_name))
