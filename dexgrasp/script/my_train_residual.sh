CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandResidualGrasp \
--algo=residual \
--seed=1 \
--rl_device=cuda:5 \
--sim_device=cuda:5 \
--logdir=logs/residual/6/no_res/test \
--base_obs_num=153 \
--num_envs=11000 \
--max_iterations=20000 \
--residual_obs_num=88 \
--headless \
--base_model_list_dir=6_means.yaml \
#--model_dir=logs/residual/6/goal_cond/test_seed1/model_20000.pt \
# --model_dir=model/MoE/5-means/delta_reward.pt \
# --test \
