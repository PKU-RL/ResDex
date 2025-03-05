CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandPcl \
--algo=dagger \
--seed=0 \
--rl_device=cuda:7 \
--sim_device=cuda:7 \
--logdir=logs/vision/4/seed2/test \
--num_envs=11000 \
--max_iteration=8000 \
--vision \
--headless \
--backbone_type pn \
#--model_dir=logs/test_seed0/model_1000.pt \
#--test
