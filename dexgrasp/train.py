# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random
import yaml
import os
import json
import sys

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex

from tqdm import tqdm

def load_yaml(file_path):   
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def save_yaml(file_path, data):
    with open(file_path, 'w') as stream:
        try:
            yaml.dump(data, stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(2)

def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["ppo", "dagger", "dagger_value", "residual"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)
    
        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations
        if args.test_all_object:
            success_tensor = None
            success_tensor = sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
            # for every entry of success_tensor, compute the final success rate
            # for each object
            # check if success_rate.yaml exists
            success_rate = {}
           
            task = sarl.vec_env.task
            # print(success_tensor.tolist())
            average_success_rate = 0
            for i in range(success_tensor.shape[0]):
                id = task.object_id_buf[i]
                object_code = task.object_code_list[id]
                object_scale = task.object_scale_buf[i]
                key = object_code+"_"+str(object_scale)
                rate = success_tensor[i].item()/200.0
                average_success_rate += rate
                success_rate[key] = rate 

            # save the result with the model name
            save_yaml(args.model_dir + "_success_rate.yaml", success_rate)
            # show how many objects have been tested
            print("Number of objects tested: ", len(success_rate.keys()))
            # calculate the average success rate
            average_success_rate /= success_tensor.shape[0]
            print("Average success rate: ", average_success_rate)
                
        else:
            sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    else:
        print("Unrecognized algorithm!")


if __name__ == '__main__':
    # print(sys.argv)
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
