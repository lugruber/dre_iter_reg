from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from data import GMDClassif, BreastCancer
from sklearn import datasets, cluster
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import argparse
from scipy import stats
from models_torch import IterLogi, IterSquare, IterCpeExp, IteratedKulsif
from models import PlattScaler, nmse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.utils.estimator_checks import check_estimator
from scipy.spatial.distance import cdist
from itertools import product


arg_parser = argparse.ArgumentParser(description=r'Set parameters for DRE')
arg_parser.add_argument(r'--dataset', type=str, help=r'name of the dataset', required=True)
arg_parser.add_argument(r'--seed', type=int,help=r'seed used for experiment', default=1)
arg_parser.add_argument(r'--nreplicates', type=int,help=r'number of geometric datasets', default=10)
arg_parser.add_argument(r'--nsamples', type=int,help=r'number of samples', default=5000)
arg_parser.add_argument(r'--ncomps', type=int,help=r'number of components', default=4)
arg_parser.add_argument(r'--dim', type=int,help=r'dimension of probability space', default=50)
arg_parser.add_argument(r'--kernel', type=str, help=r'type of kernel', default='rbf')
arg_parser.add_argument(r'--niter_iterkulsif',
                        type=int,help=r'number of iterations for Iterated KulSIF', default=1)
arg_parser.add_argument(r'--nreg_par_settings', type=int,
                        help=r'number of regularization parameter settings in experiment', default=1)
args = arg_parser.parse_args()


np.random.seed(args.seed)
if args.kernel =='rbf':
    kernel = rbf_kernel

if args.dataset == 'gmd':
    int_start, int_end = -0.1, 1
    test_set = np.random.uniform(low=int_start, high=int_end, size=(args.nsamples, args.dim))

if args.dataset == 'breast':
    dataset = BreastCancer()
    data_X, data_y = dataset.get_Xy()
    scaler = StandardScaler()
    dataset_X_stand = scaler.fit_transform(X=data_X)

    distances = np.triu(cdist(XA=dataset_X_stand, XB=dataset_X_stand))
    med = np.median(distances[distances > 0])
    rbf_gam = 1 / med
    Platt = PlattScaler(rbf_gam=rbf_gam)

    Platt.fit(X=dataset_X_stand, y=data_y)
    # [p(y=0|x), p(y=1|x)]
    pred_probas = Platt.predict_probas(dataset_X_stand)
    #reassign labels in dataset
    data_y = Platt.predict(dset=dataset_X_stand)
    # Source (y=1), Target(y=0)
    # Compute density ratios from probas: target / source
    data_Xy = np.concatenate((data_X, data_y.reshape(-1, 1)), axis=1)
    dens_rats = (pred_probas[:, 0] / pred_probas[:, 1] * (data_Xy[data_Xy[:, -1] == 1].shape[0] /
                 data_Xy[data_Xy[:, -1] == 0].shape[0]))

if args.dataset == 'gmd':
    best_scores = np.zeros(shape=(8, args.nreplicates, 10))
    # outer loop over different datasets    
    for rep in range(args.nreplicates):
	comp_means = np.random.uniform(low=int_start + 0.3, high=int_end - 0.3, size=(args.ncomps, args.dim))
	comp_stds = np.random.uniform(low=0.7, high=1., size=(args.ncomps, args.dim))
	if args.dim > 1:
	    eigs = np.random.uniform(low=1.5, high=2.5, size=(args.ncomps, args.dim))
	    eigs = eigs / eigs.sum(axis=1).reshape(-1, 1) * eigs.shape[1]
	    comp_covmats = [np.diag(comp_std) @ stats.random_correlation.rvs(eigs=eig) @ np.diag(comp_std)
	                    for comp_std, eig in zip(comp_stds, eigs)]
	else:
	    # square of this will be variance of random variables
	    comp_covmats = comp_stds
	comp_weights = np.random.uniform(low=0.1, high=1., size=args.ncomps)
	comp_weights /= comp_weights.sum()
	dataset = GMDClassif(nsamples=args.nsamples, comps=zip(comp_means, comp_covmats), comp_weights=comp_weights)
	for s in range(10):
		source_X, target_X = dataset.sample()

		# source gets pseudo-label y=1, target y=0
		source_Xy = np.concatenate((source_X, np.ones(source_X.shape[0]).reshape(-1, 1)), axis=1)
		target_Xy = np.concatenate((target_X, np.zeros(target_X.shape[0]).reshape(-1, 1)), axis=1)
		data_Xy = np.concatenate((source_Xy, target_Xy), axis=0)
		np.random.shuffle(data_Xy)
		dens_rats = dataset.get_ratio(dset=data_Xy[:, :-1])
		data_Xy_train, data_Xy_test, dens_rats_train, dens_rats_test = train_test_split(
		    data_Xy, dens_rats, test_size=0.2, shuffle=True, stratify=data_Xy[:, -1])
		param_grid_kul = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float)))}

		param_grid_iterkul = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                      'niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}
		param_grid_log = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                   'niter': list(np.arange(start=1, step=1, stop=2, dtype=float))}
		param_grid_cpeexp = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                   'niter': list(np.arange(start=1, step=1, stop=2, dtype=float))}
		param_grid_square = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                     'niter': list(np.arange(start=1, step=1, stop=2, dtype=float))}
		param_grid_itersquare = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                     'niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}
		param_grid_iterlog = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                     'niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}
		param_grid_itercpeexp = {'reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
		                     'niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}

		train_idx_h, val_idx_h = train_test_split(np.arange(data_Xy_train[:, :-1].shape[0]),
		                                      test_size=0.2, stratify=data_Xy_train[:, -1], shuffle=True)

		distances = np.triu(cdist(XA=data_Xy_train[:, :-1], XB=data_Xy_train[:, :-1]))
		med = np.median(distances[distances > 0])
		rbf_gam = 1 / med

		clf_kul = GridSearchCV(estimator=IteratedKulsif(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                       param_grid=param_grid_kul,
		                       cv=[(train_idx_h, val_idx_h)],
		                       scoring='neg_root_mean_squared_error',
		                       n_jobs=-1)
		clf_iterkul = GridSearchCV(estimator=IteratedKulsif(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                        param_grid=param_grid_iterkul,
		                        cv=[(train_idx_h, val_idx_h)],
		                        scoring='neg_root_mean_squared_error',
		                        n_jobs=-1)
		clf_log = GridSearchCV(estimator=IterLogi(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                        param_grid=param_grid_log,
		                        cv=[(train_idx_h, val_idx_h)],
		                        scoring='neg_root_mean_squared_error',
		                        n_jobs=-1)
		clf_iterlog = GridSearchCV(estimator=IterLogi(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                        param_grid=param_grid_iterlog,
		                        cv=[(train_idx_h, val_idx_h)],
		                        scoring='neg_root_mean_squared_error',
		                        n_jobs=-1)		
		clf_cpeexp = GridSearchCV(estimator=IterCpeExp(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                          param_grid=param_grid_cpeexp,
		                          cv=[(train_idx_h, val_idx_h)],
		                          scoring='neg_root_mean_squared_error',
		                          n_jobs=-1)
		clf_itercpeexp = GridSearchCV(estimator=IterCpeExp(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                          param_grid=param_grid_itercpeexp,
		                          cv=[(train_idx_h, val_idx_h)],
		                          scoring='neg_root_mean_squared_error',
		                          n_jobs=-1)		                          
		clf_square = GridSearchCV(estimator=IterSquare(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                          param_grid=param_grid_square,
		                          cv=[(train_idx_h, val_idx_h)],
		                          scoring='neg_root_mean_squared_error',
		                          n_jobs=-1)
		clf_itersquare = GridSearchCV(estimator=IterSquare(kernel=rbf_kernel, rbf_gam=rbf_gam),
		                          param_grid=param_grid_itersquare,
		                          cv=[(train_idx_h, val_idx_h)],
		                          scoring='neg_root_mean_squared_error',
		                          n_jobs=-1)
		clf_kul.fit(X=data_Xy_train, y=dens_rats_train)
		clf_iterkul.fit(X=data_Xy_train, y=dens_rats_train)
		clf_log.fit(X=data_Xy_train, y=dens_rats_train)
		clf_iterlog.fit(X=data_Xy_train, y=dens_rats_train)
		clf_cpeexp.fit(X=data_Xy_train, y=dens_rats_train)
		clf_itercpeexp.fit(X=data_Xy_train, y=dens_rats_train)
		clf_square.fit(X=data_Xy_train, y=dens_rats_train)
		clf_itersquare.fit(X=data_Xy_train, y=dens_rats_train)
		best_model_kul = clf_kul.best_estimator_
		best_model_iterkul = clf_iterkul.best_estimator_
		best_model_log = clf_log.best_estimator_
		best_model_iterlog = clf_iterlog.best_estimator_
		best_model_cpeexp = clf_cpeexp.best_estimator_
		best_model_itercpeexp = clf_itercpeexp.best_estimator_
		best_model_square = clf_square.best_estimator_
		best_model_itersquare = clf_itersquare.best_estimator_
		preds_kul = best_model_kul.predict(data_Xy_test)
		preds_iterkul = best_model_iterkul.predict(data_Xy_test)
		preds_log = best_model_log.predict(data_Xy_test)
		preds_iterlog = best_model_iterlog.predict(data_Xy_test)
		preds_cpeexp = best_model_cpeexp.predict(data_Xy_test)
		preds_itercpeexp = best_model_itercpeexp.predict(data_Xy_test)
		preds_square = best_model_square.predict(data_Xy_test)
		preds_itersquare = best_model_itersquare.predict(data_Xy_test)
		best_scores[:, rep, s] = (nmse(y_true=dens_rats_test, y_pred=preds_kul),
		                            nmse(y_true=dens_rats_test, y_pred=preds_iterkul),
		                            nmse(y_true=dens_rats_test, y_pred=preds_log),
		                            nmse(y_true=dens_rats_test, y_pred=preds_iterlog),
		                            nmse(y_true=dens_rats_test, y_pred=preds_cpeexp),
		                            nmse(y_true=dens_rats_test, y_pred=preds_itercpeexp),
		                            nmse(y_true=dens_rats_test, y_pred=preds_square),
		                            nmse(y_true=dens_rats_test, y_pred=preds_itersquare)
		                            )
    np.save(file='./results_gmd_baseline_iterexp', arr=best_scores)


if args.dataset == 'breast':
    # Outer cv loop for risk estimation
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
    param_grid_kul = {'iteratedkulsif__reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float)))}
    param_grid_iterkul = {'iteratedkulsif__reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
                          'iteratedkulsif__niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}
    param_grid_log = {'iterlogi__reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
                          'iterlogi__niter': list(np.arange(start=1, step=1, stop=2, dtype=float))}
    param_grid_cpeexp = {'itercpeexp__reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
                       'itercpeexp__niter': list(np.arange(start=1, step=1, stop=11, dtype=float))}
    param_grid_square = {'itersquare__reg_par': list(10 ** (-np.arange(start=-4, step=1, stop=7, dtype=float))),
                         'itersquare__niter': list(np.arange(start=1, step=1, stop=2, dtype=float))}
    best_scores = np.zeros(shape=(4, 10))

    loop_idx = 0
    for train_idx, test_idx in outer_cv.split(X=data_Xy[:, :-1], y=data_Xy[:, -1]):
        # Standardize for each loop
        dens_rats_train, dens_rats_test = dens_rats[train_idx], dens_rats[test_idx]
        data_Xy_train, data_Xy_test = data_Xy[train_idx], data_Xy[test_idx]

        scaler = StandardScaler()
        snt_X = scaler.fit_transform(X=data_Xy_train[:, :-1])
        distances = np.triu(cdist(XA=snt_X, XB=snt_X))
        med = np.median(distances[distances > 0])
        rbf_gam = 1 / med

        scaler = ColumnTransformer([('stand1', StandardScaler(), slice(0, data_Xy.shape[1] - 1))],
                              remainder='passthrough')
        pipeline_kul = make_pipeline(scaler, IteratedKulsif(kernel=rbf_kernel, rbf_gam=rbf_gam))
        pipeline_log = make_pipeline(scaler, IterLogi(kernel=rbf_kernel, rbf_gam=rbf_gam))
        pipeline_cpeexp = make_pipeline(scaler, IterCpeExp(kernel=rbf_kernel, rbf_gam=rbf_gam))
        pipeline_square = make_pipeline(scaler, IterSquare(kernel=rbf_kernel, rbf_gam=rbf_gam))

        # Inner cv loop for hyperparameter selection, add selection according to Bregman loss
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
        splits = list(inner_cv.split(X=data_Xy_train[:, :-1], y=data_Xy_train[:, -1]))
        train_idx_h, val_idx_h = train_test_split(np.arange(data_Xy_train[:, :-1].shape[0]),
                                              test_size=0.2, stratify=data_Xy_train[:, -1], shuffle=True)
        clf_kul = GridSearchCV(estimator=pipeline_kul, param_grid=param_grid_kul,
                               cv=[(train_idx_h, val_idx_h)],
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1
                               )
        clf_log = GridSearchCV(estimator=pipeline_log, param_grid=param_grid_log,
                                cv=[(train_idx_h, val_idx_h)],
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1
                                )
        clf_cpeexp = GridSearchCV(estimator=pipeline_cpeexp, param_grid=param_grid_cpeexp,
                                  cv=[(train_idx_h, val_idx_h)],
                                  scoring='neg_root_mean_squared_error',
                                  n_jobs=-1)
        clf_square = GridSearchCV(estimator=pipeline_square, param_grid=param_grid_square,
                                  cv=[(train_idx_h, val_idx_h)],
                                  scoring='neg_root_mean_squared_error',
                                  n_jobs=-1)
        clf_kul.fit(X=data_Xy_train, y=dens_rats_train)
        clf_log.fit(X=data_Xy_train, y=dens_rats_train)
        clf_cpeexp.fit(X=data_Xy_train, y=dens_rats_train)
        clf_square.fit(X=data_Xy_train, y=dens_rats_train)
        best_model_kul = clf_kul.best_estimator_
        best_model_log = clf_log.best_estimator_
        best_model_cpeexp = clf_cpeexp.best_estimator_
        best_model_square = clf_square.best_estimator_
        preds_kul = best_model_kul.predict(data_Xy_test)
        preds_log = best_model_log.predict(data_Xy_test)
        preds_cpeexp = best_model_cpeexp.predict(data_Xy_test)
        preds_square = best_model_square.predict(data_Xy_test)
        best_scores[:, loop_idx] = (nmse(y_true=dens_rats_test, y_pred=preds_kul),
                                    nmse(y_true=dens_rats_test, y_pred=preds_log),
                                    nmse(y_true=dens_rats_test, y_pred=preds_cpeexp),
                                    nmse(y_true=dens_rats_test, y_pred=preds_square),
                                    )
        loop_idx += 1
    np.save(file='./results_breast_baseline', arr=best_scores)

