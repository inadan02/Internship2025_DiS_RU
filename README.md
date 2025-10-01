# Towards Backdoor Stealthiness in Model Parameter Space

## Environment settings
check requirements.txt

## Train a parameter space backdoor model

```
python train_backdoor.py --pr 0.1
```
`pr` is the poisoning rate of the target class.



## Evaluate using CLP
```
python defense_lipschitzness_pruning.py
```

## Generate UPGD 
```
python generate_upgd.py --model_path ${clean_model_weights}
```
