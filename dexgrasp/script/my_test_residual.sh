CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandResidualGrasp \
--algo=residual \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--base_obs_num=153 \
--num_envs=10000 \
--max_iterations=20000 \
--base_model_list_dir=1_means.yaml \
--residual_obs_num=88 \
--headless \
--test \
--model_dir=logs/residual/6/no_res/test_seed1/model_20000.pt \
# --model_dir=model/MoE/5-means/delta_reward.pt \
