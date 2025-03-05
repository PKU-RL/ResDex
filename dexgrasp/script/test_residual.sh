CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandResidualGrasp \
--algo=residual \
--seed=8 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--num_envs=100 \
--base_model_list_dir=4_means.yaml \
--logdir=logs/test \
--base_obs_num=153 \
--residual_obs_num=88 \
--test \
--model_dir=checkpoints/res/4/state_model.pt \
# --headless \
# --save_test_traj \
# --model_dir=logs/test_seed0/model_10000.pt \
# --model_dir=model/MoE/with_objpos/fooditem_and_pencil.pt
# --model_dir=model/MoE/with_objpos/with_grasp_pose/2_plus_all_5000.pt