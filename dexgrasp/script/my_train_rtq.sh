CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandBlindGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:7 \
--sim_device=cuda:7 \
--logdir=logs/rtq/4/3/test \
--max_iterations=5000 \
--num_envs=4096 \
--headless \
# --model_dir=logs/test_seed0/model_5000.pt \
# --test \
# --model_dir=logs/test_seed0/drinking2.pt \
