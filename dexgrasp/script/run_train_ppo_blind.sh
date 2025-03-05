CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandBlindGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--max_iterations=5000 \
--num_envs=4096 \
--headless \
# --model_dir=logs/test_seed0/model_5000.pt \
# --test \
# --model_dir=logs/test_seed0/drinking2.pt \