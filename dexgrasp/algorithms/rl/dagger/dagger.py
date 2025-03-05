from datetime import datetime
from importlib.resources import path
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time

from matplotlib.patches import FancyArrow
from gym import spaces

from gym.spaces import Space

import numpy as np
import statistics
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algorithms.rl.dagger import RolloutStorage

import yaml



class DAGGER:

    def __init__(self,
                 vec_env,
                 actor_class,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 buffer_size,
                 init_noise_std=1.0,
                 learning_rate=1e-3,
                 schedule="fixed",
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 expert_chkpt_path = "",
                 is_vision = False,
                 pointnet_dir=None,
                 ):

        self.expert_chkpt_path = expert_chkpt_path
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.schedule = schedule
        self.step_size = learning_rate

        # DAGGER components
        self.buffer_size = buffer_size
        self.vec_env = vec_env
        self.actor = actor_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric, use_pc = is_vision)
        self.actor.to(self.device)

        if pointnet_dir is not None and pointnet_dir != '':
            self.actor.backbone.load_state_dict(torch.load(pointnet_dir, map_location=self.device))
            print("load pretrained model for pointnet")



        ################### create 5 base models and 1 residual hyper model ##################
        base_model_list_dir = "4_means.yaml" 
        base_model_list = yaml.load(open(base_model_list_dir, 'r'), Loader=yaml.FullLoader)["base_model_list"]
        self.base_models=[]
        for base_model_dir in base_model_list:
            base_model = actor_critic_class([153],  self.state_space.shape, self.action_space.shape,
                                                  init_noise_std, model_cfg, asymmetric=asymmetric, use_pc=False)
            base_model.to(self.device)
            base_model.load_state_dict(torch.load(base_model_dir,map_location=self.device))
            base_model.eval()
            for param in base_model.parameters():
                param.requires_grad = False
            self.base_models.append(base_model)
            print("#"*10, "base model loaded from: ", base_model_dir, "#"*10)
        
        residual_model_dir = "logs/residual/4/lift/test_seed2/model_20000.pt"
        self.residual_model = actor_critic_class([227],self.state_space.shape, [len(self.base_models)+24],
                                                    init_noise_std, model_cfg, asymmetric=asymmetric, use_pc=False)
        self.residual_model.to(self.device)
        self.residual_model.load_state_dict(torch.load(residual_model_dir, map_location=self.device))
        self.residual_model.eval()
        for param in self.residual_model.parameters():
                param.requires_grad = False
        self.softmax = nn.Softmax(dim=-1)
        


        self.storage = RolloutStorage(self.vec_env.num_envs, self.buffer_size, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # DAGGER parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        # Map any CUDA device to the available device(s)
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda(0)  # Map to the first available CUDA device
        else:
            map_location = 'cpu'  # Map to CPU if CUDA is not available
        self.actor.load_state_dict(torch.load(path, map_location=map_location))
        self.actor.eval()
       
    def load(self, path):
        self.actor.load_state_dict(torch.load(path), map_location=self.device)
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def save_pointnet(self, path):
        torch.save(self.actor.backbone.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        id = -1
        if self.is_testing:
            self.vec_env.task.random_time = False
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            length1=self.vec_env.task.max_episode_length
            for _ in range(length1):
                with torch.no_grad():
                    # Compute the action
                    id = (id+1)%self.vec_env.task.max_episode_length
                    actions = self.actor.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions,id)
                    current_obs.copy_(next_obs)
                if _ == length1-2:
                    success_rate=self.vec_env.task.successes.sum()/self.vec_env.num_envs
            print("success_rate:",success_rate)
            
        else:
            

            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    id = (id+1)%self.vec_env.task.max_episode_length
                    
                    actions = self.actor.act_inference(current_obs)
                    actions_expert = self.expert_act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions, id)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, actions_expert, rews, dones)
                    current_obs.copy_(next_obs)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _ = self.actor.act_inference(current_obs)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                mean_policy_loss = self.update()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                    self.save_pointnet(os.path.join(self.log_dir, 'pointnet_model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        self.writer.add_scalar('Loss/policy', locs['mean_policy_loss'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_policy_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]

                actions_expert_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]

                actions_batch = self.actor.act(obs_batch)

                # Policy loss
                loss = F.mse_loss(actions_batch, actions_expert_batch)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_policy_loss += loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        
        return mean_policy_loss

    def expert_act_inference(self, obs):

        def get_base_actions(obs):
            actions = []
            for base_model in self.base_models:
                actions.append(base_model.act_inference(obs).detach())
            return actions

        def compose_actions(base_actions, residual_action):
            actions = torch.zeros_like(base_actions[0])
            coef = residual_action[:, :len(self.base_models)]
            residual_action = residual_action[:, len(self.base_models):]        
            coef = self.softmax(coef)
            # print(coef)
            for i in range(len(base_actions)):
                actions += base_actions[i] * coef[:, i].unsqueeze(1)
            
            return actions + residual_action
        
        return compose_actions(get_base_actions(obs), self.residual_model.act_inference(obs).detach())
