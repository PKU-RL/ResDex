CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandBlindGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--test \
--num_envs=4096 \
--headless \
--model_dir=model/base_model/5-means/sem/ToiletPaper-accc37f006d7409d42428dd46e9da8_0.06.pt \
