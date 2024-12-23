import gym
import numpy as np
import cv2
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from script_agent.script_agent import SCRIPT_AGENTS
from script_agent.base import BaseScriptAgent
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES
from overcookedgym.overcooked_utils import LAYOUT_LIST

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from models import NNet
from tqdm import tqdm
import numpy as np
from stable_baselines3 import PPO
# import mlflow
# mlflow.autolog()

class QdrAgent():
    def __init__(self, qdr) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = qdr
        self.action_dim = qdr.action_dim

    def get_action(self, state, action_epsilon=0.0):
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = self.q_network(state).to(self.device)
                action = torch.argmax(q_values).item()
        return action

def get_dr(env, ac_dim):
    rewards = np.zeros(ac_dim)
    p = env.partners[0][0].get_action(env.mdp, env.base_env.state, 1)
    for a in range(ac_dim):
        rewards[a] = env.step_dr(actions=np.array([a, p]))
    return np.mean(rewards)


def compute_q_dr():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    num_runs = 1
    episodes = 1_000
    gamma = 0.95
    base_model_dir = f"experiments/ra-l/{layout}/"
    lr = 3e-05
    prefs = ['place_tomato_in_pot']#, 'place_onion_in_pot', 'deliver_soup', 'put_dish_everywhere', 'put_onion_everywhere', 'put_tomato_everywhere']
    env_str="OvercookedMultiEnv-v0"

    for i in range(num_runs):
        print("beginning replicate " + str(i))
        model_dir = base_model_dir #+ f"run{i}/"
        
        for p in prefs:
            print("computing q dr for " + p)
            env = gym.make(env_str, layout_name=layout)
            partner = SCRIPT_AGENTS[p]()
            env.add_partner_agent(partner)
            env.reset()
            learner = PPO.load(model_dir + f"ppo-with-{p}/model")
            #print(env.observation_space.shape[0], env.lA)
            qdr = NNet(env.observation_space.shape[0], env.lA, 1)
            qdr_optim = optim.AdamW(qdr.parameters(), lr=1e-4, amsgrad=True)
            egreedy = 0.99
            erate = 0.999
            emin = 0.1
            for ep in tqdm(range(episodes), desc="agent-rollouts"):
                obs = env.reset()
                info = {}
                done = False
                while not done:
                    # get agent action
                    if np.random.rand() < egreedy:
                        action = env.action_space.sample()
                    else:
                        action, _ = learner.predict(obs, deterministic=True)
                    egreedy = max(emin, egreedy * erate)

                    dr = get_dr(env, env.lA)

                    # step env with selected action
                    obs_next, rew, done, info_next = env.step(action)

                    target = rew - dr
                    next_action, _ = learner.predict(obs_next, deterministic=True)
                    action = torch.from_numpy(np.array([action])).view(-1)
                    next_action = torch.from_numpy(np.array([next_action])).view(-1)
                    #print(torch.from_numpy(obs).unsqueeze(0).shape)
                    cur_q = qdr(torch.from_numpy(obs).unsqueeze(0)).squeeze(-1)[0, action]
                    next_q = target + gamma * qdr(torch.from_numpy(obs_next).unsqueeze(0)).squeeze(-1)[0, next_action]
                    loss = nn.MSELoss()(cur_q.float(), next_q.float())
                    
                    qdr_optim.zero_grad()
                    loss.backward()
                    qdr_optim.step()

                    if done:
                        break
                    obs = obs_next
                    info = info_next
            torch.save(qdr, model_dir + f"{p}_qdr.torch")

if __name__ == "__main__":
    compute_q_dr()