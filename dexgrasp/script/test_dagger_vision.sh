CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandPcl \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--vision \
--backbone_type pn \
--model_dir=logs/vision/goal_lift/4/model_6000.pt \
--test \
--pointnet_dir=logs/vision/goal_lift/4/pointnet_model_6000.pt \
--num_envs=1 \
# --headless \
# --num_envs=11000 \
