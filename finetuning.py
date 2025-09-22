import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
import cv2
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy

from script_agent.script_agent import SCRIPT_AGENTS, Place_Tomato_and_Deliver_Soup_Agent
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES
from overcookedgym.overcooked_utils import LAYOUT_LIST
import wandb

layout = 'simple_o_t'
assert layout in LAYOUT_LIST
num_runs = 1
# 'place_tomato_in_pot', 790, 710, 720, 490, 500, 165
partner_types = ['place_onion_in_pot', 'deliver_soup', 'put_dish_everywhere', 'put_onion_everywhere', 'put_tomato_everywhere']
episodes = 1_000

for r in range(num_runs):
    for p in partner_types:
        # Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
        # register an environment and construct it using gym.make.
        env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)
        wandb.tensorboard.unpatch()

        tensorboard_dir=f"experiments/ijcai/{layout}/ppo-with-{p}/"
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
        wandb.init(
            # set the wandb project where this run will be logged
            project="ppo-finetuning",

            # track hyperparameters and run metadata
            config={
                "layout": layout,
                "episodes": episodes,
                "partner_type": p,
                "run_num": r,
            }
        )

        # Before training your ego agent, you first need to add your partner agents
        # to the environment.
        partner = SCRIPT_AGENTS[p]()
        env.unwrapped.add_partner_agent(partner)
        env.reset()

        # Finally, you can construct an ego agent and train it in the environment
        # ego = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_dir+f"run{r}/")
        model_dir = f"experiments/ijcai/{layout}/"
        ego = PPO.load(model_dir + f"ppo-with-{p}/model", env=env)
        mean_reward, std_reward = evaluate_policy(ego, env, n_eval_episodes=10)
        print("ppo mean rew: ", mean_reward, " and std rew: ", std_reward)
        # Switch to train mode (this affects batch norm / dropout)
        ego.policy.set_training_mode(True)

        for ep in tqdm(range(episodes), desc="ppo-finetuning"):
            obs, _ = env.reset()
            info = {}
            done = False
            ep_rew = 0.0
            ep_dr = 0.0
            ep_loss = 0.0
            while not done:
                action, _states = ego.predict(obs)
                obs, r, terminated, truncated, info = env.step(action)
                ep_rew += r
                done = terminated or truncated

                obs_tensor = obs.reshape((-1,) + ego.observation_space.shape)
                obs_tensor = obs_as_tensor(obs_tensor, ego.device)
                val, log_prob, ent = ego.policy.evaluate_actions(obs_tensor, torch.from_numpy(action))
                #print(r, val)
                #print(info)
                ego_r = info['sparse_r_by_agent'][0] + info['shaped_r_by_agent'][0]
                ep_dr += ego_r
                #print(ego_r, torch.tensor([ego_r]).float().shape, val.item(), val.shape)
                ego_r_tensor = torch.tensor([ego_r]).float()
                val_loss = F.mse_loss(ego_r_tensor, val[0])
                loss = ego.vf_coef * val_loss
                ep_loss += loss.item()
                # Optimization step
                ego.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(ego.policy.parameters(), ego.max_grad_norm)
                ego.policy.optimizer.step()
            wandb.log({"reward": ep_rew, "dr": ep_dr, "loss": ep_loss})

        ego.save(tensorboard_dir+"finetuned")
        ego = PPO.load(tensorboard_dir+"finetuned", env=env)
        # for ep in tqdm(range(10), desc="finetuning-eval"):
        #     obs, _ = env.reset()
        #     info = {}
        #     done = False
        #     ep_rew = 0.0
        #     while not done:
        #         action, _states = ego.predict(obs)
        #         obs, r, terminated, truncated, info = env.step(action)
        #         ep_rew += r
        #         done = terminated or truncated

        #         obs_tensor = obs.reshape((-1,) + ego.observation_space.shape)
        #         obs_tensor = obs_as_tensor(obs_tensor, ego.device)
        #         val, log_prob, ent = ego.policy.evaluate_actions(obs_tensor, torch.from_numpy(action))

        #     wandb.log({"tuned/reward": ep_rew})

        mean_reward, std_reward = evaluate_policy(ego, env, n_eval_episodes=10)
        print("ppo tuned mean rew: ", mean_reward, " and std rew: ", std_reward)