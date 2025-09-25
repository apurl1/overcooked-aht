import warnings
import os
from typing import Any, ClassVar, Optional, Union, TypeVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import cv2

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule, GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

from script_agent.script_agent import SCRIPT_AGENTS
from overcookedgym.overcooked_utils import LAYOUT_LIST
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES

import wandb

PHI_EVENT_TYPES = [
    "potting_tomato",
    "potting_onion",
    "useful_dish_pickup",
    "soup_pickup",
    "soup_delivery",
]
PHI_DIM = len(PHI_EVENT_TYPES)

def phi(info_cur, info_next):
    phi_vec = np.zeros(len(PHI_EVENT_TYPES))
    for i, et in enumerate(PHI_EVENT_TYPES):
        if info_cur:
            phi_old = len(info_cur[et][0]) + len(info_cur[et][1])
        else:
            phi_old = 0
        phi_new = len(info_next[et][0]) + len(info_next[et][1])
        phi_vec[i] = phi_new - phi_old
    return phi_vec

class PsiNetwork(BasePolicy):
    """
    Psi network for SFDQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        w: np.ndarray = np.ones(PHI_DIM),
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim

        action_dim = int(self.action_space.n)
        self.psi_net = nn.ModuleList(
            [nn.Sequential(*create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)) for i in range(PHI_DIM)]
        )
        self.w = torch.from_numpy(w).float()
    
    def get_w(self) -> torch.Tensor:
        return self.w
    
    def set_w(self, new_w: torch.Tensor) -> None:
        self.w = new_w

    def q_vals(self, obs: PyTorchObs) -> torch.Tensor:
        psi_values = self(obs)
        # print(psi_values.shape, self.w.shape)
        q_values = torch.matmul(psi_values.float(), self.w.float())
        return q_values
    
    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Predict the psi values.

        :param obs: Observation
        :return: The estimated value for each action.
        """
        obs = self.extract_features(obs, self.features_extractor)
        outputs = [self.psi_net[i](obs).unsqueeze(-1) for i in range(PHI_DIM)]
        return torch.cat(outputs, dim=-1)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        psi_values = self(observation)
        q_values = torch.matmul(psi_values.float(), self.w.float())
        # Greedy action
        greedy_action = q_values.argmax(dim=1).reshape(-1)
        return greedy_action

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class SFPolicy(BasePolicy):
    psi_net: PsiNetwork
    psi_net_target: PsiNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        w: np.ndarray = np.ones(PHI_DIM),
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.w = w

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "w": self.w,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.psi_net = self.make_psi_net()
        self.psi_net_target = self.make_psi_net()
        self.psi_net_target.load_state_dict(self.psi_net.state_dict())
        self.psi_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.psi_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_psi_net(self) -> PsiNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return PsiNetwork(**net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        return self.psi_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return super()._get_constructor_parameters()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.psi_net.set_training_mode(mode)
        self.training = mode

SelfSFDQN = TypeVar("SelfSFDQN", bound="SFDQN")

class SFDQN(OffPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {"MlpPolicy": SFPolicy,}
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    psi_net: PsiNetwork
    psi_net_target: PsiNetwork
    policy: SFPolicy

    def __init__(
        self,
        policy: Union[str, type[SFPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.95,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.psi_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.psi_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.psi_net = self.policy.psi_net
        self.psi_net_target = self.policy.psi_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.psi_net.parameters(), self.psi_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next values using the target network
                next_psi_values = self.psi_net_target(replay_data.next_observations)
                next_q_values = self.psi_net_target.q_vals(replay_data.next_observations)
                next_actions = torch.argmax(next_q_values, dim=1).unsqueeze(-1)
                next_psi_values = next_psi_values.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, PHI_DIM)).squeeze(1)
                # 1-step TD target
                # print(replay_data.rewards.shape, next_psi_values.shape)
                target_psi_values = replay_data.rewards.reshape(batch_size, PHI_DIM) + (1 - replay_data.dones) * self.gamma * next_psi_values

            # Get current value estimates
            current_psi_values = self.psi_net(replay_data.observations)

            # Retrieve the values for the actions from the replay buffer
            current_psi_values = torch.gather(current_psi_values, dim=1, index=replay_data.actions.long().unsqueeze(-1).expand(-1, -1, PHI_DIM)).squeeze(1)

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_psi_values, target_psi_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfSFDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSFDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "psi_net", "psi_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def train_ego_agent():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    num_runs = 10
    partner_types = ['deliver_soup', 'pickup_tomato_and_place_mix', 'pickup_onion_and_place_mix']
    gamma = 0.95
    episodes = 2_500_000
    w_team = np.array([3.0, 3.0, 3.0, 5.0, 20.0])

    for r in range(num_runs):
        for p in partner_types:
            env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
            tensorboard_dir=f"experiments/aaai/{layout}/sfdqn-with-{p}/run{r}/"
            os.makedirs(tensorboard_dir, exist_ok=True)
            
            wandb.init(
                project="sfdqn-training-overcooked",
                sync_tensorboard=True,
                config={
                    "layout": layout,
                    "timesteps": episodes,
                    "gamma": 0.95,
                    "partner_type": p,
                    "run_num": r,
                }
            )

            partner = SCRIPT_AGENTS[p]()
            env.unwrapped.add_partner_agent(partner)
            env.reset()
            env = Monitor(env, tensorboard_dir)

            ego = SFDQN('MlpPolicy', env, tensorboard_log=tensorboard_dir, verbose=0, gamma=gamma)
            
            # Create the callback: check every 10,000 steps
            callback = SaveOnBestTrainingRewardCallback(check_freq=25_000, log_dir=tensorboard_dir)
            ego.psi_net.set_w(torch.from_numpy(w_team))
            ego.psi_net_target.set_w(torch.from_numpy(w_team))
            ego.learn(total_timesteps=episodes, progress_bar=True, callback=callback)

            # generate video of a rollout with the trained agents
            obs, _ = env.reset()
            done = False
            vvw = cv2.VideoWriter(tensorboard_dir+f"rollout.mp4", cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))

            while not done:
                action, _states = ego.predict(obs)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                vvw.write(env.render())
                game_stats = env.unwrapped.get_game_stats()
                wandb.log({"agent1/sparse": game_stats["cumulative_sparse_rewards_by_agent"][0],
                            "agent2/sparse": game_stats["cumulative_sparse_rewards_by_agent"][1],
                            "agent1/shaped": game_stats["cumulative_shaped_rewards_by_agent"][0],
                            "agent2/shaped": game_stats["cumulative_shaped_rewards_by_agent"][1],
                            "team/shaped": np.sum(game_stats["cumulative_shaped_rewards_by_agent"]),
                            "team/sparse": np.sum(game_stats["cumulative_sparse_rewards_by_agent"])})
                for et in EVENT_TYPES:
                    wandb.log({f"agent1/{et}": len(game_stats[et][0])})
                    wandb.log({f"agent2/{et}": len(game_stats[et][1])})

            ego.save(tensorboard_dir+ "model")

if __name__ == "__main__":
    train_ego_agent()