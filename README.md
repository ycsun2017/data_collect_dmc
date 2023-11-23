# Collecting Data in DMC

## Install
```
conda create -n dmc python=3.9
pip install -r requirements.txt
cd dmc2gym &&  pip install -e . && cd ../
```

## Run data collection

1. Collect by training an SAC agent (with vector state input):
```
# format:
bash scripts/collect.sh $GAME $TASK $SEED $NAME
# example:
bash scripts/collect.sh walker walk 123 walker_walk_sac_s123
```

2. Collect by a random policy:
```
# format:
bash scripts/collect_rand.sh $GAME $TASK $SEED $NAME
# example:
bash scripts/collect_rand.sh walker walk 123 walker_walk_rand_s123
```

This codebase is modified from [SAC-AE](https://sites.google.com/view/sac-ae/home)
