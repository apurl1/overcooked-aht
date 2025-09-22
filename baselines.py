"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

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
        # if self.num_timesteps % 399 == 0:
        #     self.logger.dump(self.num_timesteps)
        return True

layout = 'simple_o_t'
assert layout in LAYOUT_LIST
num_runs = 1
partner_types = ['pickup_onion_and_place_mix', 'pickup_tomato_and_place_mix']
tensorboard_dir=f"experiments/aaai/{layout}/robust/"
episodes = 2_000_000

for r in range(num_runs):
    for i, p in enumerate(partner_types):
        env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
        #wandb.tensorboard.unpatch()
        #wandb.tensorboard.patch(root_logdir=tensorboard_dir)
        wandb.init(
            # set the wandb project where this run will be logged
            project="robust-learner-overcooked",
            sync_tensorboard=True,

            # track hyperparameters and run metadata
            config={
                "layout": layout,
                "timesteps": episodes,
                "partner_type": p,
                "run_num": r,
            }
        )

        partner = SCRIPT_AGENTS[p]()
        env.unwrapped.add_partner_agent(partner)
        env.reset()
        if i == 0:
            ego = SFDQN('MlpPolicy', env, tensorboard_log=tensorboard_dir, verbose=0)
        else:
            ego = SFDQN.load(tensorboard_dir + "model")
            ego.set_env(env)
        print('training with ' + p)
        ego.learn(total_timesteps=episodes, progress_bar=True, callback=TensorboardCallback())
        ego.save(tensorboard_dir+ "model")
