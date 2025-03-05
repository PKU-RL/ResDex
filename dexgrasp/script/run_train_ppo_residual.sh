CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandResidualGrasp \
--algo=residual \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--base_obs_num=153 \
--num_envs=11000 \
--max_iterations=20000 \
--base_model_list_dir=model/base_model_list.yaml \
--residual_obs_num=88 \
--headless \
# --model_dir=logs/test_seed0/model_10000.pt \
# --model_dir=model/MoE/5-means/delta_reward.pt \
# --test \
