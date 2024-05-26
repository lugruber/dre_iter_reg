import sys
sys.path.append('..')
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import argparse
from scipy import stats
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist
from trainer import cross_domain_dre

arg_parser = argparse.ArgumentParser(description=r'Set parameters for DRE')
arg_parser.add_argument(r'--debug', type=bool, help=r'flag for debugging mode', default=True)
arg_parser.add_argument(r'--dataset', type=str, help=r'name of the dataset', required=True)
arg_parser.add_argument(r'--save_dir', type=str, help=r'directory to save iwv dictionaries',
                        default='../datasets/')
arg_parser.add_argument(r'--data_path', type=str, help=r'path to load data',
                        default='../datasets/original/')
arg_parser.add_argument(r'--da_method', type=str, help=r'name of the domain adaptation method',
                        required=True)
arg_parser.add_argument(r'--backbone', type=str, help=r'type of backbone architecture',
                        default='CNN')
arg_parser.add_argument(r'--device', type=str, help=r'computation device', default='cpu')
arg_parser.add_argument(r'--seed', type=int, help=r'seed used for experiment', default=1)
arg_parser.add_argument(r'--iwv_method', type=str,
                        help=r'method for computing importance weights', default='IWV_Domain_Classifier')
arg_parser.add_argument(r'--dre_method', type=str, required=True)
arg_parser.add_argument(r'--num_runs', type=int, help=r'number of replicates', default=3)
arg_parser.add_argument(r'--experiment_name', type=str, default='multirun')
arg_parser.add_argument(r'--run_description', type=str, default='da')



arg_parser.add_argument(r'--kernel', type=str, help=r'type of kernel', default='rbf')
arg_parser.add_argument(r'--niter_iterkulsif',
                        type=int, help=r'number of iterations for Iterated KulSIF', default=1)
arg_parser.add_argument(r'--nreg_par_settings', type=int,
                        help=r'number of regularization parameter settings in experiment', default=1)
arg_parser.add_argument(r'--reg_par',
                        type=float, help=r'exponent regularization parameter for Iterated KulSIF', default=3)
                        
args = arg_parser.parse_args()

#np.random.seed(args.seed)
if args.kernel =='rbf':
    kernel = rbf_kernel

DRE = cross_domain_dre(args=args)
DRE.train()


