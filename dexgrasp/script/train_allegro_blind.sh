CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=AllegroHandBlindGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--max_iterations=5000 \
--num_envs=4096 \
# --num_envs=100 \
# --headless \
# --model_dir=logs/test_seed0/model_5000.pt \
# --test \
# --model_dir=logs/test_seed0/drinking2.pt \