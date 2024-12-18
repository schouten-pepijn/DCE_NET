import torch


class network_training_hyper_parameters:
    def __init__(self):
        self.lr = 1e-4
        self.lr_mult = 0.1
        self.epochs = 70  # was 50
        self.optim = 'adam'  # adam 0.0001; sgd 0.1
        self.patience = self.epochs
        self.optim_patience = 3  # 0 is disabled
        self.batch_size = 256  # alias = N
        self.val_batch_size = 1280
        self.split = 0.9
        self.totalit = 1000
        self.save_train_fig = True
        self.weight_decay = 0


class acquisition_parameters:
    def __init__(self):
        self.timing = None

class network_building_hyper_parameters:
    def __init__(self):
        self.pretrained = False
        self.dropout = 0
        self.nn = 'gru'  # ['linear', 'convlin', 'lstm']
        self.layers = [32, 4]
        self.attention = True


class simulation_hyper_parameters:
    def __init__(self):
        self.what_to_sim = "nn"  # T1fit, lsq or nn
        self.plot = True
        self.bounds = torch.FloatTensor(((1e-6, 1e-6, 1e-6, -1.),
                                         (2., 1.5, 0.2, 2.)
                                         ))  # ke, ve, vp, T1, dt, ((min), (max))


class Hyperparams:
    def __init__(self):
        '''Hyperparameters'''
        # main
        self.acquisition = acquisition_parameters()
        self.training = network_training_hyper_parameters()
        self.network = network_building_hyper_parameters()
        self.simulations = simulation_hyper_parameters()
        self.use_cuda = True
        self.exp_name = 'pretrained_gru_net'     

        # Preprocessing
        self.Hct = 0.4
        self.dt_timing = 4.8  # seconds
        self.acquisition_points = 65
        self.bolus_time = 24. / 60.  # min
        self.relaxivity = 4.3
        self.n_jobs = 12
        self.remove_slices = True
        self.remove_bounds = (2, 15)
        self.global_threshold = True
        self.enhance = True
        self.pop_aif = True
        
        # training parameters
        self.mask_zeros = True
        self.mask_cutoff = 0.05
        self.mask_extremes = True
        self.mask_val = (-0.5, 1)
        self.full_interference = True
        self.aif_type = 'pop'  # (pop or pat)
        self.t0_value = 0.
        
        # dirs
        self.clin_ids = (('Clinical_P1', 'Visit1'),
                         ('Clinical_P1', 'Visit2'),
                         ('Clinical_P2', 'Visit1'),
                         ('Clinical_P2', 'Visit2'),
                         ('Clinical_P3', 'Visit1'),
                         ('Clinical_P3', 'Visit2'),
                         ('Clinical_P4', 'Visit1'),
                         ('Clinical_P4', 'Visit2'),
                         ('Clinical_P5', 'Visit1'),
                         ('Clinical_P5', 'Visit2'),
                         ('Clinical_P6', 'Visit1'),
                         ('Clinical_P6', 'Visit2'),
                         ('Clinical_P7', 'Visit1'),
                         ('Clinical_P7', 'Visit2'),
                         ('Clinical_P8', 'Visit1'),
                         ('Clinical_P8', 'Visit2'))
        self.synth_ids = (('Synthetic_P1', 'Visit1'),
                          ('Synthetic_P1', 'Visit2'),
                          ('Synthetic_P2', 'Visit1'),
                          ('Synthetic_P2', 'Visit2'))
        self.flip_ids = (' 5 flip', '10 flip', '15 flip',
                         '20 flip', '25 flip', '30 flip')
        self.folder_ids = ('net_output', 'results', 'training')
