CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandGrasp \
--algo=residual \
--seed=0 \
--rl_device=cuda:7 \
--sim_device=cuda:7 \
--logdir=logs/fullobs_MoE/5/5/test \
--base_obs_num=300 \
--num_envs=11000 \
--max_iterations=20000 \
--residual_obs_num=88 \
--headless \
--base_model_list_dir=fullobs_5_means.yaml \
# --model_dir=logs/test_seed0/model_10000.pt \
# --model_dir=model/MoE/5-means/delta_reward.pt \
# --test \
