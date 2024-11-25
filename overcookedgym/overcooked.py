import gym

import cv2
import pygame
import copy
import numpy as np
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcookedgym.overcooked_ai.src.overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcookedgym.overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from pantheonrl.common.multiagentenv import SimultaneousEnv

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        mlap = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.base_env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlap)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx

        self.visualizer = StateVisualizer()
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)
    
    def get_game_stats(self):
        return self.base_env.game_stats

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r_by_agent'][self.ego_agent_idx]
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs), (reward, reward), done, info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        for player in self._players:
            if player != self.ego_ind:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                agent.reset(self.mdp, self.base_env.state, p)

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self.base_env.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self.visualizer.render_state(
            state=self.base_env.state,
            grid=self.base_env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(
                self.base_env.state, **rewards_dict
            ),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image

    def _get_actions(self, players, obs, ego_act=None):
        actions = []
        for player, ob in zip(players, obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                actions.append(agent.get_action(self.mdp, self.base_env.state, player))
                if not self.should_update[p]:
                    agent.update(self.total_rews[player], False)
                self.should_update[p] = True
        return np.array(actions)
