CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--model_dir=logs/test_seed0/original_grasp.pt \
--test \
--headless \