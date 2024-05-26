# Overcoming Saturation in Densitiy Ratio Estimation by Iterated Regularization

## Requirmenets:
- Python3
- Numpy==1.21.2
- scikit-learn==1.0.2
- Pandas==1.4.2
- Pytorch==1.11.0
- Hydra=1.2.0
- Wandb=0.12.7
- skorch==0.10.0
- OmegaConf=2.2.3
- openpyxl==3.0.7

## Domain Adaptation
In directory `Exp_Aggregation`.
Train domain adaptation methods with `run_da.py`, example runs in `example_runs.sh`.
Train density ratio estimation methods with `rund_dre.py`.
Evaluate with `get_results.py`.

## Density Ratio Estimation
In directory `Exp_Kanamori`.
