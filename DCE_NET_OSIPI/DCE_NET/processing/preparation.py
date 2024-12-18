import os
import numpy as np
import time
import pydicom
import math
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from tkinter import Tk
from tkinter.filedialog import askdirectory

class load():
    @classmethod
    def dicom_wrapper(cls, data, flip_ids=None):
        # for FA paths
        tic = time.time()
        if flip_ids is not None:
            print('reading FA images..')
            pat_keys = sorted(data.keys())
            flip_keys = np.unique([
                flip for key in pat_keys for flip in flip_ids])
            ds_top = ()
            for pat_key in pat_keys:
                ds_sub = ()
                for flip_key in flip_keys:
                    path = data[pat_key][flip_key]
                    ds_sub += (cls.load_scan(path),)
                ds_top += (ds_sub,)
            ds_top = np.array(ds_top)
            pixel_array = np.array(
                [[cls.get_pixels(ds2) for ds2 in ds1] for ds1 in ds_top])
            pixel_array = np.swapaxes(pixel_array, 0, 1)
            shape = pixel_array.shape

        # for perfusion paths
        else:
            print('reading DCE images..')
            # get dicom data
            pat_keys = sorted(data.keys())
            ds_top = ()
            for pat_key in pat_keys:
                path = data[pat_key]
                ds_top += (cls.load_scan(path),)
            ds_top = np.array(ds_top)
            pixel_array = np.array([cls.get_pixels(ds) for ds in ds_top])
            pixel_array = pixel_array.reshape(len(ds_top), 65, 16, 256, 256)
            pixel_array = np.swapaxes(pixel_array, 0, 1)
            shape = pixel_array.shape
            ds_top = ds_top.reshape(len(ds_top), 65, 16)

        params = cls.get_params(ds_top)
        toc = time.time()
        print(f'load time: {toc - tic}')
        return pixel_array, params, shape

    def load_scan(path):
        slices = [pydicom.dcmread(os.path.join(path, s),
                                  force=False) for s in os.listdir(path)]
        # slices = [s for s in slices if 'SliceLocation' in s]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices

    def get_pixels(scans, air_zero=False, hu=False, normalize=False):
        image = np.stack(
            [s.pixel_array for s in scans]).astype(dtype=np.float64)
        return image

    def get_params(ds):
        print('getting scan parameters')

        FA = np.zeros((ds.shape[0], ds.shape[1]))
        TR = np.zeros_like(FA)
        TE = np.zeros_like(FA)
        for i in range(ds.shape[0]):
            for j in range(ds.shape[1]):
                FA[i, j] = math.radians(ds[i, j][0].FlipAngle)  # radians
                TR[i, j] = ds[i, j][0].RepetitionTime * 1e-3  # ms
                TE[i, j] = ds[i, j][0].EchoTime * 1e-3  # ms
        FA = np.unique(FA)
        TR = np.unique(TR)
        TE = np.unique(TE)
        params = {'FA': FA, 'TR': TR, 'TE': TE}
        return params
    
    def read_paths(main_dir, clin_ids, synth_ids, flip_ids):
        tags = ('Clinical_Data', 'Synthetic_Data')
        print('reading file paths') 
        clin_ids = [os.path.join(tags[0], clin_num, visit_num) for (clin_num, visit_num) in clin_ids]
        synth_ids = [os.path.join(tags[1], synth_num, visit_num) for (synth_num, visit_num) in synth_ids]
        paths_clin = [os.path.join(root, dirr) for root, dirs, _ in os.walk(
            main_dir, topdown=True) for dirr in dirs if 'Clinical' in root]
        paths_synth = [os.path.join(root, dirr) for root, dirs, _ in os.walk(
            main_dir, topdown=True) for dirr in dirs if 'Synthetic' in root]
        
        clin_ids = [idd[len('Clinical_Data/'):] for idd in clin_ids]
        synth_ids = [idd[len('Synthetic_Data/'):] for idd in synth_ids]
        
        dce_clin_dirs = sorted(
            [path for path in paths_clin if 'perfusion' in path])
        dce_synth_dirs = sorted(
            [path for path in paths_synth if 'perfusion' in path])
        fa_clin_dirs = sorted(
            [path for path in paths_clin if 'flip' in path])
        fa_synth_dirs = sorted(
            [path for path in paths_synth if 'flip' in path])
    
        # DCE clinical images
        dce_clin = {ID.lower(): path for ID, path in zip(
            clin_ids, dce_clin_dirs) if ID in path}
        assert len(clin_ids) == len(dce_clin_dirs)
        
        # DCE synthetic images
        dce_synth = {ID.lower(): path for ID, path in zip(
            synth_ids, dce_synth_dirs) if ID in path}
        assert len(synth_ids) == len(dce_synth_dirs)
    
        # T1 FA clinical images
        fa_clin = {}
        for ID in clin_ids:
            temp = {}
            for path in fa_clin_dirs:
                if ID in path:
                    for flip in flip_ids:
                        if flip in path:
                            temp[flip] = path
            fa_clin[ID.lower()] = temp
    
        # T1 FA synthetic images
        fa_synth = {}
        for ID in synth_ids:
            temp = {}
            for path in fa_synth_dirs:
                if ID in path:
                    for flip in flip_ids:
                        if flip in path:
                            temp[flip] = path
            fa_synth[ID.lower()] = temp
        return fa_clin, fa_synth, dce_clin, dce_synth

    def get_timing(dt, num, minute=True):
        print('getting scan timing')
        t = np.linspace(0, dt*num, num)
        if minute:
            t /= 60
        return t


class preprocessing():
    def global_threshold(img_dce, img_fa, cutoff=1.3):
        print('applying global threshold..')
        mask_th = np.zeros((len(img_fa), *img_dce.shape[3:],))
        for k, (array_fa, array_dce) in enumerate(zip(img_fa, img_dce)):
            mask = (np.mean(array_fa, axis=(0, 1))
                    > np.median(array_fa)*cutoff)
            mask_th[k] = mask
            array_dce[:, :, ~mask] = 0
            img_dce[k] = array_dce
            array_fa[:, :, ~mask] = 0
            img_fa[k] = array_fa

        return mask_th, img_dce, img_fa


class postprocessing():
    def remove_nan(data, value=0):
        return np.nan_to_num(data, nan=value)

    def enhance(signal, time, bolus_time, sds=1, percentage=30):
        print('selecting enhancing voxels')
        delay = np.abs(time - bolus_time).argmin()
        stds = np.std(signal[:delay, :], axis=0)
        means = np.mean(signal[:delay, :], axis=0)
        cutoff = np.array(means+sds*stds)
        selects = signal[delay:, :] < np.repeat(cutoff[np.newaxis],
                                                len(signal[delay:, :]),
                                                axis=0)
        selected = np.sum(selects, axis=0) < percentage/100*(np.shape(
            signal)[0]-delay)
        signal[:, ~selected] = 0
        return selected, signal

    def delete_slices(data, shape_in, i_min, i_max):
        output = data.reshape(*shape_in)
        output = output[:, :, i_min:i_max]
        shape_out = output.shape
        output = output.reshape(len(data), -1)
        return output, shape_out


class convert():
    def calc_S0(images, t, bolus_time, idx_margin=None):
        # image shape: voxels x time
        print('calculating S0 map')
        images = images.reshape(65, -1).T
        idx = np.argwhere(t < bolus_time).squeeze()
        if idx_margin is not None:
            idx_margin = 1 if idx.max() < idx_margin else idx_margin
            idx = idx[:-idx_margin]
        S0 = np.mean(images[:, idx], axis=1)
        return S0

    def fit_R1(images, params, cutoff, n_jobs=12):
        ''' Create T1 map from multiflip images '''
        flip_angles = sorted([angle for angle in params['FA']])
        TR = params['TR']
        images = images.reshape(len(images), -1).T
        inshape = images.shape
        nangles = inshape[-1]
        n = np.prod(inshape[:-1])
        images = np.reshape(images, (n, nangles))
        assert(nangles == len(flip_angles))
        signal_scale = abs(images).max()
        images = images / signal_scale
    
        def t1_signal_eqn(x, M0, R1):
            E1 = np.exp(-TR*R1)
            return M0*np.sin(x)*(1.0 - E1) / (1.0 - E1*np.cos(x))
        
        def parfun(j):
            if images[j,:].mean() > 0.05:
                try:
                    (S0, R1), _ = curve_fit(t1_signal_eqn, flip_angles, 
                                           images[j,:], bounds=(0, np.inf))
                except RuntimeError:
                    S0 = 0
                    R1 = 0
            else:
                S0 = 0
                R1 = 0
            return S0, R1

        output = Parallel(n_jobs=n_jobs, verbose=1000)(
            delayed(parfun)(i) for i in range(n))
        output = np.array(output)
        S0map = output[:, 0]
        R10map = output[:, 1]
        S0map *= signal_scale
        return R10map, S0map 

    def dce_to_r1eff(S, S0, R1, params):
            print ('converting DCE signal to effective R1')
            flip = params['FA']
            TR = params['TR']
            S = S.reshape(len(S), -1)
            R1 = R1.T
            S0 = S0.T
            R1map = []
            for S_i in S:
                S_i = S_i.T
                A = S_i / S0  # normalize by pre-contrast signal
                E0 = np.exp(-R1 * TR)
                E = (1.0 - A + A*E0 - E0*np.cos(flip)) /\
                     (1.0 - A*np.cos(flip) + A*E0*np.cos(flip) - E0*np.cos(flip))
                R = (-1.0 / TR) * np.log(E)
                R1map.append(R)
            R1map = np.array(R1map)
            return R1map

    def r1eff_to_conc(R1eff, R1map, relaxivity):
        print ('converting effective R1 to tracer tissue concentration')
        Cmap = []
        for R1eff_i in R1eff:
            C = (R1eff_i - R1map) / relaxivity
            Cmap.append(C)
        Cmap = np.array(Cmap)
        return Cmap


class aif():
    def aifpopHN(Hct=0.4):
        # defines plasma curve; note this is a population-based AIF for H&N
        # patients (https://doi.org/10.2967/jnumed.116.174433). 
        aif = {'ab': 3.646/(1-Hct), 'mb': 25.5671,
                'ae': 1.53, 'me': 0.2130, 't0': 0.1}
        return aif


def main(hp):
    # load the paths to the dicom series from the folder structure
    try:
        window = Tk()
        main_dir = askdirectory(title='Select OSIPI data folder')
        if main_dir == ():
            main_dir = askdirectory(title='Select OSIPI data folder')
        window.quit()
        window.destroy()
    except:
        while True:
            main_dir = input('Give OSIPI data folder path:')
            if os.path.exists(main_dir):
                break

    # load file paths
    fa_clin, fa_synth, dce_clin, dce_synth = load.read_paths(
        main_dir, hp.clin_ids, hp.synth_ids, hp.flip_ids)

    # load the pixel arrays from the T1 FLASH and DCE data
    img_fa_clin, params_fa_clin, _ = load.dicom_wrapper(
        fa_clin, hp.flip_ids)
    # img_fa_synth, params_fa_synth, shape_fa_synth = load.dicom_wrapper(
    #     fa_synth, hp.flip_ids)
    img_dce_clin, params_dce_clin, shape_dce_clin = load.dicom_wrapper(
        dce_clin)
    # img_dce_synth, params_dce_synth, shape_dce_synth = load.dicom_wrapper(
    #     dce_synth)

    del fa_clin, dce_clin

    # generate the acquisition timing
    timing = load.get_timing(dt=hp.dt_timing,
                             num=hp.acquisition_points,
                             minute=True)

    # calculate the R10 map of the clinical data based on multiple flip angles
    R10map_clin, _ = convert.fit_R1(img_fa_clin,
                                    params_fa_clin,
                                    hp.mask_cutoff,
                                    n_jobs=hp.n_jobs)

    del img_fa_clin, params_fa_clin

    # calculate the baseline signal of the clinical data
    S0map_clin = convert.calc_S0(img_dce_clin,
                                  timing,
                                  hp.bolus_time,
                                  idx_margin=2)

    # calculate the effective R1 signal of the clinical data
    R1effmap_clin = convert.dce_to_r1eff(img_dce_clin,
                                          S0map_clin,
                                          R10map_clin,
                                          params_dce_clin)

    del img_dce_clin, params_dce_clin, 
    del S0map_clin

    # calculate the concentration map of the clinical data
    Cmap_clin = convert.r1eff_to_conc(R1effmap_clin,
                                      R10map_clin,
                                      hp.relaxivity)

    del R10map_clin, R1effmap_clin

    # remove NaN values
    Cmap_clin = postprocessing.remove_nan(Cmap_clin, 0)

    # select enhancing voxels in clinical data
    if hp.enhance:
        _, Cmap_clin_enh = postprocessing.enhance(Cmap_clin,
                                                  timing,
                                                  hp.bolus_time)

    # remove uninformative slices in the clinical data
    if hp.remove_slices:
        Cmap_clin_enh, shape_enh_clin = postprocessing.delete_slices(Cmap_clin_enh,
                                                                     shape_dce_clin,
                                                                     hp.remove_bounds[0],
                                                                     hp.remove_bounds[1])
    
    # load the pixel arrays from the T1 FLASH and DCE data
    img_fa_synth, params_fa_synth, _ = load.dicom_wrapper(
        fa_synth, hp.flip_ids)
    img_dce_synth, params_dce_synth, shape_dce_synth = load.dicom_wrapper(
        dce_synth)

    del fa_synth, dce_synth

    # calculate R10 map of the syntethic data based on multiple flip angles
    R10map_synth, _ = convert.fit_R1(img_fa_synth,
                                      params_fa_synth,
                                      hp.mask_cutoff,
                                      n_jobs=hp.n_jobs)

    del img_fa_synth, params_fa_synth

    # calculate baseline signal of the synthetic data
    S0map_synth = convert.calc_S0(img_dce_synth,
                                  timing,
                                  hp.bolus_time,
                                  idx_margin=2)

    # calculate effective R1 map of synthetic data
    R1effmap_synth = convert.dce_to_r1eff(img_dce_synth,
                                        S0map_synth,
                                        R10map_synth,
                                        params_dce_synth)

    del img_dce_synth, params_dce_synth
    del S0map_synth

    # calculate concentration map of synthetic data
    Cmap_synth = convert.r1eff_to_conc(R1effmap_synth,
                                        R10map_synth,
                                        hp.relaxivity)
    
    del R1effmap_synth
    
    # remove NaN values
    Cmap_synth = postprocessing.remove_nan(Cmap_synth, 0)

    # select enhancing voxels in synthetical data
    if hp.enhance:
            _, Cmap_synth_enh = postprocessing.enhance(Cmap_synth,
                                                        timing,
                                                        hp.bolus_time)

    # remove uninformative slices in synthetic data
    if hp.remove_slices:
        Cmap_synth_enh, shape_enh_synth = postprocessing.delete_slices(Cmap_synth_enh,
                                                                       shape_dce_synth,
                                                                       hp.remove_bounds[0],
                                                                       hp.remove_bounds[1])

    # concatenate clinical and synthetic data into one training data set
    dce_train = np.concatenate((
        Cmap_clin_enh.reshape(*shape_enh_clin),
        Cmap_synth_enh.reshape(*shape_enh_synth)), axis=1)
    dce_train_shape = dce_train.shape
    dce_train = dce_train.reshape(65, -1).T

    del Cmap_clin_enh, Cmap_synth_enh

    # concatenate clinical and synthetic data into one testing data set
    dce_test = np.concatenate((
        Cmap_clin.reshape(*shape_dce_clin),
        Cmap_synth.reshape(*shape_dce_synth)), axis=1)
    dce_test_shape = dce_test.shape
    dce_test = dce_test.reshape(65, -1).T

    del Cmap_clin, Cmap_synth

    # retrieve the population based AIF data
    aif_params = aif.aifpopHN(hp.Hct)

    # output data
    out = (timing,
           dce_train, dce_train_shape,
           dce_test, dce_test_shape,
           aif_params)
    return out