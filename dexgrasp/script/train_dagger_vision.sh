CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandPcl \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/vision/4/seed2/test \
--num_envs=10 \
--max_iteration=8000 \
--vision \
--backbone_type pn \
# --headless \
