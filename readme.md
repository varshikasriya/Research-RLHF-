smoke test command line:
python smorm_train.py --smoke --model Qwen/Qwen2.5-0.5B --out runs/smoke

real test command line: 
python smorm_train.py --model google/gemma-2b-it --w1 1 --w2 1 --out runs/w1_1_w2_1_seed42

set up on a gpu machine:
pip install -r requirements.txt    
huggingface-cli login 
#login for gated models like google/gemma-2b-it
