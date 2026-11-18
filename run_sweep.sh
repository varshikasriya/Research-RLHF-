
#!/usr/bin/env bash
set -e
MODEL="google/gemma-2b-it"
EXTRA="--bs 4 --accum 4 --grad_ckpt"
PAIRS=("1 1" "0.5 0.5" "0.25 0.75" "0.75 0.25" "0.9 0.1" "0.1 0.9" "1 0" "0 1")
SEEDS=(42 123 456)

for seed in "${SEEDS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    set -- $pair
    out="runs/w1_$1_w2_$2_seed$seed"
    if [ -f "$out/results.json" ]; then echo "skip $out"; continue; fi
    python smorm_train.py --model "$MODEL" --w1 "$1" --w2 "$2" --seed "$seed" $EXTRA --out "$out"
  done
done

python make_table.py