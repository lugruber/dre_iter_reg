# runs to train domain adaptation methods

PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=MINI_DOMAIN_NET backbone=Pretrained2D da_method=AdvSKM
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=AMAZON_REVIEWS backbone=MLP da_method=CDAN
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=CoDATS
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DANN
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DIRT
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DSAN
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=DDC
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=HoMM
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=Deep_Coral
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=CoDATS
PYTHONPATH=. python run_da.py -m seed=1 experiment_name=multirun run_description=da dataset=HHAR_SA da_method=MMDA

