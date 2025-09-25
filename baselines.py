"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import os
import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from script_agent.script_agent import SCRIPT_AGENTS, Place_Tomato_and_Deliver_Soup_Agent
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES
from overcookedgym.overcooked_utils import LAYOUT_LIST
from sfdqn import SFDQN
import wandb

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        game_stats = self.training_env.env_method("get_game_stats")[0]
        self.logger.record("agent1/sparse", game_stats["cumulative_sparse_rewards_by_agent"][0])
        self.logger.record("agent2/sparse", game_stats["cumulative_sparse_rewards_by_agent"][1])
        self.logger.record("agent1/shaped", game_stats["cumulative_shaped_rewards_by_agent"][0])
        self.logger.record("agent2/shaped", game_stats["cumulative_shaped_rewards_by_agent"][1])
        self.logger.record("team/shaped", np.sum(game_stats["cumulative_shaped_rewards_by_agent"]))
        self.logger.record("team/sparse", np.sum(game_stats["cumulative_sparse_rewards_by_agent"]))
        for et in EVENT_TYPES:
            self.logger.record(f"agent1/{et}", len(game_stats[et][0]))
            self.logger.record(f"agent2/{et}", len(game_stats[et][1]))
        return True

layout = 'simple_o_t'
assert layout in LAYOUT_LIST
num_runs = 10
partner_types = ['pickup_onion_and_place_mix', 'pickup_tomato_and_place_mix']
episodes = 2_500_000 * len(partner_types)

for r in range(num_runs):
    tensorboard_dir=f"experiments/aaai/{layout}/robust_test/run{r}/"
    os.makedirs(tensorboard_dir, exist_ok=True)
    env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
    
    wandb.init(
        project="robust-learner-overcooked",
        sync_tensorboard=True,

        config={
            "layout": layout,
            "timesteps": episodes,
            "run_num": r,
        }
    )

    for p in partner_types:
        partner = SCRIPT_AGENTS[p]()
        env.unwrapped.add_partner_agent(partner)
    env.reset()
    ego = SFDQN('MlpPolicy', env, tensorboard_log=tensorboard_dir, verbose=0)
    ego.learn(total_timesteps=episodes, progress_bar=True, callback=TensorboardCallback())
    ego.save(tensorboard_dir+ "model")
