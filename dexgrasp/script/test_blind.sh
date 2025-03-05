CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandBlindGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:1 \
--sim_device=cuda:1 \
--logdir=logs/test \
--test \
--num_envs=11000 \
--headless \
--model_dir=logs/2/0/test_seed0/model_20000.pt \
#--test_all_object \
# --model_dir=logs/test_seed0/model_5000.pt \
# --model_dir=model/base_model/with_objpos/cellphone_5000.pt \
# --model_dir=model/base_model/fooditem_5000.pt \
# --model_dir=model/base_model/with_objpos/pencil_5000.pt \
