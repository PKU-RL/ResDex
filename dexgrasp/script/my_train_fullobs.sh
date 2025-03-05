CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/fullobs/4/3/test \
--max_iterations=5000 \
--num_envs=4096 \
--headless \
# --model_dir=example_model/model.pt \
#--test
