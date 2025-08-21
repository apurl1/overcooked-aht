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
partner_types = ['pickup_onion_and_place_mix', 'deliver_soup']
episodes = 1_000_000

for r in range(num_runs):
    for p in partner_types:
        # Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
        # register an environment and construct it using gym.make.
        env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
        #wandb.tensorboard.unpatch()
        tensorboard_dir=f"experiments/aaai/{layout}/dqn-lossless-with-{p}/"
        #wandb.tensorboard.patch(root_logdir=tensorboard_dir)
        wandb.init(
            # set the wandb project where this run will be logged
            project="dqn-lossless-training-overcooked",
            sync_tensorboard=True,

            # track hyperparameters and run metadata
            config={
                "layout": layout,
                "timesteps": episodes,
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
        #ego = DQN('MlpPolicy', env, tensorboard_log=tensorboard_dir, verbose=0)
        ego = DQN('MlpPolicy', env, tensorboard_log=tensorboard_dir, verbose=0)
        ego.learn(total_timesteps=episodes, progress_bar=True, callback=TensorboardCallback())

        # generate video of a rollout with the trained agents
        obs, info = env.reset()
        done = False
        vvw = cv2.VideoWriter(tensorboard_dir+f"rollout.mp4", cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))
        frames = []

        while not done:
            action, _states = ego.predict(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            vvw.write(env.render())

        ego.save(tensorboard_dir+ "model")
