
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'iwv_epochs': 150,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'lambdas': [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10],
        }
        self.alg_hparams = {
            'DANN':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 0.9238},
            'Deep_Coral':   {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'DDC':          {'learning_rate': 0.01,     'src_cls_loss_wt': 0.1593,  'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'DSAN':         {'learning_rate': 0.005,    'src_cls_loss_wt': 0.4133,  'domain_loss_wt': 0.16},
            'AdvSKM':       {'learning_rate': 0.001,    'src_cls_loss_wt': 0.4637,  'domain_loss_wt': 0.1511},
            'MMDA':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,   'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
            'DIRT':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947},
            'CMD':          {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 5.5238,   'cmd_moments': 4}
        }


class MINI_DOMAIN_NET():
    def __init__(self):
        super(MINI_DOMAIN_NET, self).__init__()
        self.train_params = {
            'num_epochs': 60,
            'iwv_epochs': 100,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'lambdas': [0, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10],
        }
        self.alg_hparams = {
            'DANN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 0.9238},
            'Deep_Coral':   {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'DDC':          {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.1593,  'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'DSAN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.4133,  'domain_loss_wt': 0.16},
            'AdvSKM':       {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.4637,  'domain_loss_wt': 0.1511},
            'MMDA':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,   'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
            'DIRT':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947},
            'CMD':          {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 5.5238,   'cmd_moments': 4}
        }


class AMAZON_REVIEWS():
    def __init__(self):
        super(AMAZON_REVIEWS, self).__init__()
        self.train_params = {
            'num_epochs': 50,
            'iwv_epochs': 80,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'lambdas': [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10],
        }
        self.alg_hparams = {
            'DANN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 0.9238},
            'Deep_Coral':   {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'DDC':          {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.1593,  'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'DSAN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.4133,  'domain_loss_wt': 0.16},
            'AdvSKM':       {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.4637,  'domain_loss_wt': 0.1511},
            'MMDA':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,   'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
            'DIRT':         {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947},
            'CMD':          {'learning_rate': 1e-3,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt': 5.5238,   'cmd_moments': 4}
        }



