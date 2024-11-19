"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
import numpy as np
import cv2
from stable_baselines3 import PPO

from script_agent.script_agent import SCRIPT_AGENTS, Place_Tomato_and_Deliver_Soup_Agent
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcookedgym.overcooked_utils import LAYOUT_LIST

layout = 'simple_tomato'
assert layout in LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)

# Before training your ego agent, you first need to add your partner agents
# to the environment.
partner = SCRIPT_AGENTS['place_tomato_in_pot']()
env.add_partner_agent(partner)
env.reset()


# Finally, you can construct an ego agent and train it in the environment
tensorboard_dir="experiments/ra-l/overcooked/runs/ppo/"
ego = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
ego.learn(total_timesteps=500_000)

obs = env.reset()
done = False
vvw = cv2.VideoWriter(tensorboard_dir+'test.mp4', cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))
frames = []

while not done:
    action, _states = ego.predict(obs)
    obs, r, done, info = env.step(action)
    vvw.write(env.render())
