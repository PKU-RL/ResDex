CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandResidualGrasp \
--algo=residual \
--seed=1 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/res\
--base_obs_num=153 \
--num_envs=11000 \
--max_iterations=20000 \
--base_obs_num=153 \
--residual_obs_num=88 \
--base_model_list_dir=4_means.yaml \
# --headless \
