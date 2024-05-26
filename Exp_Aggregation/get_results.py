import sys
sys.path.append('..')
from approaches.ensemble_trainer import EnsembleTrainer

res_dirs = ['../datasets/AMAZON_REVIEWS/processed/AMAZON_REVIEWS', '../datasets/MINI_DOMAIN_NET/processed/MINI_DOMAIN_NET', '../datasets/HHAR_SA/processed/HHAR_SA']
seeds = [1,2,3]
rconds = [2e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dre_methods = ['kul', 'iterkul', 'cpeexp', 'itercpeexp', 'log', 'iterlog', 'square', 'itersquare']
da_methods = ["AdvSKM", "CDAN", "CMD", "CoDATS", "DANN", "DDC", "Deep_Coral", "DIRT", "DSAN", "HoMM", "MMDA"]
extractor_dirs = ['extractor/adatime_amazon_results_loader.py', 'extractor/adatime_minidomainnet_results_loader.py', 'extractor/adatime_hhar_sa_results_loader.py']
manual_filter_lambdas = []
suffix = ''

for da_method in da_methods:
    for dre_method in dre_methods:
        for res_dir, extractor_dir in zip(res_dirs, extractor_dirs):
            for rcond in rconds:                
                et = EnsembleTrainer(base_dir=res_dir + f'/{da_method}', da_method=da_method, seed_list=seeds, extractor=extractor_dir,
                                     rcond=rcond, manual_filter_lambdas=manual_filter_lambdas, suffix=suffix, dre_method=dre_method, cv=True)
                et.run()
                
