import numpy as np
import torch
import os
import nibabel as nib

np.random.seed(42)
torch.manual_seed(42)

def mask_extremes(signal, bounds):
    low, high = bounds[0], bounds[1]
    print(f'masking extreme signal voxels < {low} and > {high}')
    print(f'shape in: {signal.shape}')
    mask_low = (signal < high).all(axis=1)
    signal = signal[mask_low, :]
    mask_high = (signal > low).all(axis=1)
    signal = signal[mask_high, :]
    print(f'shape out: {signal.shape}')
    return signal

def mask_zeros(signal, cutoff):
    print('masking zero signal voxels')
    print(f'shape in: {signal.shape}')
    mask = (signal != 0).any(axis=1)
    signal = signal[mask, :]
    if cutoff != 0:
        print(f'masking mean < {cutoff} signal voxels')
        mask = np.mean(signal, axis=1) > cutoff
        signal = signal[mask, :]
    print(f'shape out: {signal.shape}')
    return signal, mask

def aif_loader_pop(aif_dicts, t0, device=None):
    print(f'converting aif t0 value to {t0}')
    if t0 is not None:
            aif_vals = torch.FloatTensor([t0 if key == 't0'
                                          else aif_dicts[key]
                                          for key 
                                          in ('t0', 'ab', 'mb', 'ae', 'me')])
    else:
        aif_vals = torch.FloatTensor([aif_dicts[key]
                                      for key
                                      in ('t0', 'ab', 'mb', 'ae', 'me')])
    if device:
        aif_vals = aif_vals.to(device)
    return aif_vals

def param_map(params, shapes, save_path=None, save_nii=False):
    print('creating parameter maps')
    X_pred, ktrans = params
    dce_shape, param_shape = shapes
    ktrans_map = ktrans.reshape(*param_shape)
    X_pred = np.swapaxes(X_pred.T.reshape(*dce_shape), 0, 1)
    params = [X_pred, ktrans_map]
    if save_path is not None:
        if save_nii:
            array_to_nii(ktrans_map, path=save_path,
                         file_name='ktrans_map_tot.nii.gz',
                         affine=np.eye(4), transpose=True)
            for i, ktrans_i in enumerate(ktrans_map):
                if i < 16:
                    array_to_nii(ktrans_i, path=save_path,
                                 file_name=f'Clinical_P{i//2+1}_Visit{i%2+1}.nii.gz',
                                 affine=np.eye(4), transpose=True)
                else:
                    array_to_nii(ktrans_i, path=save_path,
                                 file_name=f'Synthetic_P{i//2-7}_Visit{i%2+1}.nii.gz',
                                 affine=np.eye(4), transpose=True)
    return params


def array_to_nii(data, path, file_name, affine, transpose=False):
        # x y z t
        print(f'saving {file_name} to nifti')
        if transpose:
            data = np.flip(data, 1)
            data = data.T
            
        ni_img = nib.Nifti1Image(data, affine)
        nib.save(ni_img, os.path.join(path, file_name))
        
        
def create_folder(folders):
    print('creating save folders')
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)