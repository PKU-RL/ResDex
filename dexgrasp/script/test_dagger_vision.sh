CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandPcl \
--algo=dagger \
--seed=8 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--vision \
--backbone_type pn \
--model_dir=checkpoints/vision/vision_model.pt \
--test \
--pointnet_dir=checkpoints/vision/pointnet_model.pt \
--num_envs=1 \
# --headless \
# --num_envs=11000 \
