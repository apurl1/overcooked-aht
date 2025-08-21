import gymnasium as gym
import numpy as np
import cv2
import copy
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor

from script_agent.script_agent import SCRIPT_AGENTS
from overcookedgym.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES
from overcookedgym.overcooked_utils import LAYOUT_LIST

from sfdqn import SFDQN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from models import NNet
from tqdm import tqdm
import wandb
from sklearn.linear_model import LinearRegression
# import imitation.data.rollout as rollout
# from imitation.data import serialize
# from imitation.data.huggingface_utils import trajectories_to_dataset

PHI_EVENT_TYPES = [
    "potting_tomato",
    "potting_onion",
    "useful_dish_pickup",
    "soup_pickup",
    "soup_delivery",
]

PHI_DIM = len(PHI_EVENT_TYPES)

def phi(info, info_next):
    phi_vec = np.zeros(len(PHI_EVENT_TYPES))
    for i, et in enumerate(PHI_EVENT_TYPES):
        if info:
            phi_old = len(info[et][0]) + len(info[et][1])
        else:
            phi_old = 0
        phi_new = len(info_next[et][0]) + len(info_next[et][1])
        phi_vec[i] = phi_new - phi_old
    #print(phi_vec)
    return phi_vec

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
                #print(q_values)
                action = torch.argmax(q_values).item()
                #print(action)
        return action

def get_dr(env, ac_dim):
    rewards = np.zeros(ac_dim)
    p = env.unwrapped.partners[0][0].get_action(env.unwrapped.mdp, env.unwrapped.base_env.state, 1)
    for a in range(ac_dim):
        rewards[a] = env.unwrapped.step_dr(actions=np.array([a, p]))
    return np.mean(rewards)

def get_ppo_vals():
    # set up env and load ppo
    layout = 'simple_o_t'
    env_str="OvercookedMultiEnv-v0"
    model_dir = f"experiments/ijcai/{layout}/"
    env = gym.make(env_str, layout_name=layout)
    p1 = 'place_tomato_in_pot'
    p2 = 'place_onion_in_pot'
    pnew = 'deliver_soup'
    partner = SCRIPT_AGENTS[pnew]()
    env.add_partner_agent(partner)
    env.reset()
    learner1 = PPO.load(model_dir + f"ppo-with-{p1}/model", env=env)
    learner2 = PPO.load(model_dir + f"ppo-with-{p2}/model", env=env)
    learner_new = PPO.load(model_dir + f"ppo-with-{pnew}/model", env=env)
    wandb.init(
        # set the wandb project where this run will be logged
        project="ppo-values-test",

        # track hyperparameters and run metadata
        config={
            "layout": layout,
        }
    )

    # baselines
    mean_reward, std_reward = evaluate_policy(learner1, env, n_eval_episodes=100)
    print("ppo learner1 mean rew: ", mean_reward, " and std rew: ", std_reward)
    mean_reward, std_reward = evaluate_policy(learner2, env, n_eval_episodes=100)
    print("ppo learner2 mean rew: ", mean_reward, " and std rew: ", std_reward)
    mean_reward, std_reward = evaluate_policy(learner_new, env, n_eval_episodes=100)
    print("ppo learner_new mean rew: ", mean_reward, " and std rew: ", std_reward)

    rews = []
    for ep in tqdm(range(100), desc="agent-rollouts"):
        obs, _ = env.reset()
        info = {}
        done = False
        ep_rew = 0.0
        pol = 0
        pol_counter = 0
        if ep == 99:
            vvw = cv2.VideoWriter(model_dir+"gpat_rollout.mp4", cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))
        while not done:
            obs_tensor = obs.reshape((-1,) + learner_new.observation_space.shape)
            obs_tensor = obs_as_tensor(obs_tensor, learner_new.device)
            val1 = learner1.policy.predict_values(obs_tensor)
            val2 = learner2.policy.predict_values(obs_tensor)
            if (val1 > val2 and pol_counter == 0) or (pol_counter > 0 and pol == 1):
                action, _ = learner1.predict(obs, deterministic=True)
                pol = 1
            elif (val1 <= val2 and pol_counter == 0) or (pol_counter > 0 and pol == 2):
                action, _ = learner2.predict(obs, deterministic=True)
                pol = 2

            pol_counter += 1
            if pol_counter > 5:
                pol_counter = 0
            # step env with selected action
            #action, _ = learner_new.predict(obs, deterministic=True)
            obs_next, rew, terminated, truncated, info_next = env.step(action)
            ep_rew += rew
            done = terminated or truncated
            wandb.log({"rew": rew, "policy": pol, "action": action, "val1": val1, "val2": val2})
            if done:
                break
            if ep == 99:
                vvw.write(env.render())
            obs = obs_next
            info = info_next
        #print("gpat rew: ", ep_rew)
        rews.append(ep_rew)
    rew_arr = np.array(rews)
    print("avg rew: ", np.mean(rew_arr), "std rew: ", np.std(rew_arr))

def compute_w_dr():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    num_runs = 1
    partner_types = ['pickup_tomato_and_place_mix', 'pickup_onion_and_place_mix']
    w_team = np.array([3.0, 3.0, 3.0, 5.0, 20.0])
    episodes = 50

    for r in range(num_runs):
        for p in partner_types:
            X = []
            y = []
            env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
            tensorboard_dir=f"experiments/aaai/{layout}/sfdqn-with-{p}/"

            partner = SCRIPT_AGENTS[p]()
            env.unwrapped.add_partner_agent(partner)
            env.reset()
            ego = SFDQN.load(tensorboard_dir + "model")
            ego.psi_net.set_w(torch.from_numpy(w_team))
            # print(ego.psi_net.get_w())

            for ep in tqdm(range(episodes), desc="agent-rollouts"):
                obs, info = env.reset()
                game_stats = env.unwrapped.get_game_stats()

                while True:
                    game_stats_prev = copy.deepcopy(game_stats)

                    # get agent action
                    action, _states = ego.predict(obs, deterministic=True)
                    dr = get_dr(env, 6)

                    # step env with selected action
                    obs_next, rew, terminated, truncated, info_next = env.step(action)
                    game_stats = env.unwrapped.get_game_stats()
                    #print(np.sum(rew), dr)
                    y.append(np.sum(rew) - dr)
                    X.append(phi(game_stats_prev, game_stats))

                    if terminated or truncated:
                        break
                    obs = obs_next
                    info = info_next
            w = LinearRegression().fit(X, y)
            print(w.score(X, y))
            print(w.coef_)
            np.save(tensorboard_dir + 'w-sf.npy', np.array(w.coef_))

def tune_q_net():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    partner_type = 'pickup_onion_and_place_mix'
    episodes = 10
    env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
    tensorboard_dir=f"experiments/aaai/{layout}/dqn-with-{partner_type}/"
    wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    wandb.init(
        # set the wandb project where this run will be logged
        project="dqn-w-update-overcooked",
        sync_tensorboard=True,

        # track hyperparameters and run metadata
        config={
            "layout": layout,
            "timesteps": episodes,
            "partner_type": partner_type,
        }
    )
    partner = SCRIPT_AGENTS[partner_type]()
    w = np.load(tensorboard_dir + 'w.npy')
    env.unwrapped.add_partner_agent(partner)
    env.reset()
    ego = DQN.load(tensorboard_dir + "model")
    ego.policy.set_training_mode(True)
    for ep in tqdm(range(episodes), desc="agent-rollouts"):
        obs, info = env.reset()
        game_stats = env.get_game_stats()
        ep_rew = []
        ep_loss = []
        ep_dr = []
        while True:
            game_stats_prev = copy.deepcopy(game_stats)

            # get agent action
            action, _states = ego.predict(obs, deterministic=True)

            # step env with selected action
            obs_next, rew, terminated, truncated, info_next = env.step(action)
            ep_rew += rew
            
            game_stats = env.get_game_stats()
            phi_vec = phi(game_stats_prev, game_stats)
            phi_w = np.dot(phi_vec, w)
            ep_dr += phi_w
            next_q = ego.q_net(obs_next)
            next_q = next_q.reshape(-1, 1)
            next_q, _ = next_q.max(dim=1)
            target_q = phi_w + (1 - terminated) * ego.gamma * next_q
            cur_q = ego.q_net(obs)
            cur_q = torch.gather(cur_q, dim=1, index=action)
            loss = F.smooth_l1_loss(cur_q, target_q)
            ep_loss += loss
            ego.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ego.policy.parameters(), ego.max_grad_norm)
            ego.policy.optimizer.step()

            if terminated or truncated:
                break
            obs = obs_next
            info = info_next
        wandb.log({"reward": ep_rew, "dr": ep_dr, "loss": ep_loss})

def compute_q_dr():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    num_runs = 1
    episodes = 1_000
    gamma = 0.95
    base_model_dir = f"experiments/ijcai/{layout}/"
    lr = 1e-4
    prefs = ['place_onion_in_pot']#, 'place_onion_in_pot', 'deliver_soup', 'put_dish_everywhere', 'put_onion_everywhere', 'put_tomato_everywhere']
    env_str="OvercookedMultiEnv-v0"

    wandb.init(
        # set the wandb project where this run will be logged
        project=f"qdr-training-{layout}",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "layout": layout,
            "episodes": episodes,
            "prefs": prefs,
        }
    )

    for i in range(num_runs):
        print("beginning replicate " + str(i))
        model_dir = base_model_dir #+ f"run{i}/"
        
        for p in prefs:
            print("computing q dr for " + p)
            # set up env and load ppo
            env = gym.make(env_str, layout_name=layout)
            partner = SCRIPT_AGENTS[p]()
            env.add_partner_agent(partner)
            env.reset()
            learner = PPO.load(model_dir + f"ppo-with-{p}/model", env=env)
            # instantiate qdr
            qdr = NNet(env.observation_space.shape[0], env.lA, 1)
            qdr_optim = optim.AdamW(qdr.parameters(), lr=lr, amsgrad=True)
            egreedy = 0.0
            erate = 0.99
            emin = 0.1
            # verify ppo performance
            mean_reward, std_reward = evaluate_policy(learner, learner.get_env(), n_eval_episodes=10)
            print("trained ppo learner mean rew: ", mean_reward, " and std rew: ", std_reward)
            # evaluate qdr
            for ep in tqdm(range(episodes), desc="qdr-training"):
                obs, info = env.reset()
                done = False
                ep_rew = 0.0
                ep_loss = 0.0
                t = 0
                while not done:
                    # get egreedy agent action
                    if np.random.rand() < egreedy:
                        action = env.action_space.sample()
                    else:
                        action, _ = learner.predict(obs, deterministic=True)
                    egreedy = max(emin, egreedy * erate)

                    dr = get_dr(env, env.lA)

                    # step env with selected action
                    obs_next, rew, done, truncated, info_next = env.step(action)
                    t += 1
                    #print("ppo action: ", action)
                    #print(env.mdp.state_string(env.base_env.state))
                    #print("ppo rew: ", rew)

                    target = rew - dr
                    
                    wandb.log({"rew": rew, "dr": dr, "td": target})
                    game_stats = env.get_game_stats()
                    wandb.log({"agent1/sparse": game_stats["cumulative_sparse_rewards_by_agent"][0]})
                    wandb.log({"agent2/sparse": game_stats["cumulative_sparse_rewards_by_agent"][1]})
                    wandb.log({"agent1/shaped": game_stats["cumulative_shaped_rewards_by_agent"][0]})
                    wandb.log({"agent2/shaped": game_stats["cumulative_shaped_rewards_by_agent"][1]})
                    wandb.log({"team/shaped": np.sum(game_stats["cumulative_shaped_rewards_by_agent"])})
                    wandb.log({"team/sparse": np.sum(game_stats["cumulative_sparse_rewards_by_agent"])})
                    for et in EVENT_TYPES:
                        wandb.log({f"agent1/{et}": len(game_stats[et][0])})
                        wandb.log({f"agent2/{et}": len(game_stats[et][1])})
                    
                    next_action, _ = learner.predict(obs_next, deterministic=True)
                    action = torch.from_numpy(np.array([action])).view(-1)
                    next_action = torch.from_numpy(np.array([next_action])).view(-1)
                    #print(torch.from_numpy(obs).unsqueeze(0).shape)
                    cur_q = qdr(torch.from_numpy(obs).unsqueeze(0)).squeeze(-1)[0, action]
                    next_q = target + gamma * qdr(torch.from_numpy(obs_next).unsqueeze(0)).squeeze(-1)[0, next_action]
                    loss = nn.MSELoss()(cur_q.float(), next_q.float())
                    ep_loss += loss.item()
                    ep_rew += rew
                    
                    qdr_optim.zero_grad()
                    loss.backward()
                    qdr_optim.step()

                    if done:
                        break
                    obs = obs_next
                    info = info_next
                    #done=True
                wandb.log({"ep-rew": ep_rew, "ep-loss": ep_loss / t})
            torch.save(qdr, model_dir + f"{p}_qdr.torch")
    
def evaluate_q_dr():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    episodes = 100
    prefs = ['place_onion_in_pot']#, 'place_onion_in_pot', 'deliver_soup', 'put_dish_everywhere', 'put_onion_everywhere', 'put_tomato_everywhere']
    env_str="OvercookedMultiEnv-v0"
    base_model_dir = f"experiments/ijcai/{layout}/"

    for p in prefs:
        print("evaluating q dr for " + p)
        env = gym.make(env_str, layout_name=layout)
        partner = SCRIPT_AGENTS[p]()
        env.add_partner_agent(partner)
        env.reset()
        qdr = torch.load(base_model_dir + p + "_qdr.torch")
        learner = QdrAgent(qdr)
        #print(env.observation_space.shape[0], env.lA)
        tot_rew = []
        for ep in tqdm(range(episodes), desc="agent-rollouts"):
            ep_rew = 0.0
            obs, info = env.reset()
            done = False
            while not done:
                # get agent action
                action = learner.get_action(obs)
                # step env with selected action
                obs_next, rew, done, truncated, info_next = env.step(action)
                ep_rew += rew
                if done:
                    break
                obs = obs_next
                info = info_next
            tot_rew.append(ep_rew)
        print(f"avg rew for {p}: " + str(np.mean(np.array(tot_rew))))
    # generate video of a rollout with the trained agents
    obs, info = env.reset()
    done = False
    vvw = cv2.VideoWriter(base_model_dir+f"{p}_qdr_rollout.mp4", cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))
    while not done:
        action = learner.get_action(obs)
        obs, r, done, truncated, info = env.step(action)
        vvw.write(env.render())

def get_trajs():
    layout = 'cramped_corridor'
    assert layout in LAYOUT_LIST
    episodes = 100
    base_model_dir = f"experiments/ra-l/{layout}/"
    prefs = ['place_tomato_in_pot']#, 'place_onion_in_pot', 'deliver_soup', 'put_dish_everywhere', 'put_onion_everywhere', 'put_tomato_everywhere']
    p = prefs[0]
    env_str="OvercookedMultiEnv-v0"
    env = gym.make(env_str, layout_name=layout)
    partner = SCRIPT_AGENTS[p]()
    env.add_partner_agent(partner)
    env.reset()
    learner = PPO.load(base_model_dir + f"ppo-with-{p}/model", env=env)
    trajectories = rollout.rollout(
        learner,
        env,
        sample_until=rollout.make_sample_until(min_episodes=episodes),
        rng=np.random.default_rng(),
        unwrap=False,
    )
    serialize.save(base_model_dir, trajectories)

def zero_shot():
    layout = 'simple_o_t'
    assert layout in LAYOUT_LIST
    pretrained_partners = ['pickup_onion_and_place_mix', 'pickup_tomato_and_place_mix']
    partner_type = 'deliver_soup'
    episodes = 1
    env = gym.make('OvercookedMultiEnv-v1', layout_name=layout)
    tensorboard_dir=f"experiments/aaai/{layout}/zeroshot-with-{partner_type}/"
    print(tensorboard_dir)
    wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    wandb.init(
        project="zeroshot-overcooked",
        sync_tensorboard=True,
        config={
            "layout": layout,
            "timesteps": episodes,
            "partner_type": partner_type,
        }
    )
    pretrained_agents = []
    factor = [583.83, 609.93]
    # factor = [1.0, 1.0]
    for pt in pretrained_partners:
        pt_dir = f"experiments/aaai/{layout}/sfdqn-with-{pt}/"
        w = np.load(pt_dir + 'w.npy')
        ego = SFDQN.load(pt_dir + "best_model")
        ego.psi_net.set_w(torch.from_numpy(w))
        print(ego.policy.psi_net.w)
        pretrained_agents.append(ego)
    partner = SCRIPT_AGENTS[partner_type]()
    env.unwrapped.add_partner_agent(partner)
    env.reset()
    tot_rew = []
    pol_used = np.zeros(len(pretrained_agents))
    actions_taken = np.zeros(6)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        cur_pol_steps = 0
        idx_prev = 0
        while not done:
            obs = obs.reshape((-1,) + ego.observation_space.shape)
            obs = obs_as_tensor(obs, ego.device)
            q_vals = []
            for i, ego in enumerate(pretrained_agents):
                with torch.no_grad():
                    q_vals.append(torch.matmul(ego.policy.psi_net(obs), ego.policy.psi_net.w.float()) / factor[i])
            q_vals = torch.stack(q_vals)
            loc = torch.argmax(q_vals).item()
            action = loc % 6
            idx = loc // 6
            if idx != idx_prev:
                if cur_pol_steps < 5:
                    idx = idx_prev
                    action = torch.argmax(q_vals).item() % 6
                else:
                    cur_pol_steps = 0
                cur_pol_steps += 1
            if idx == idx_prev:
                cur_pol_steps += 1
            pol_used[idx] += 1
            obs, r, terminated, truncated, info = env.step(action)
            actions_taken[action] += 1
            ep_rew += (np.sum(info['sparse_r_by_agent']) + np.sum(info['shaped_r_by_agent']))
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
            done = terminated or truncated
            idx_prev = idx
        tot_rew.append(ep_rew)
    # generate video of a rollout with the trained agents
    obs, info = env.reset()
    done = False
    vvw = cv2.VideoWriter(tensorboard_dir+f"rollout.mp4", cv2.VideoWriter_fourcc('X','V','I','D'),10,(2 * 528, 2 * 464))

    while not done:
        action, _states = ego.predict(obs)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        vvw.write(env.render())
    
    print(np.mean(np.array(tot_rew)), np.std(np.array(tot_rew)))
    print(tot_rew)
    print(pol_used)
    print(actions_taken)

if __name__ == "__main__":
    # compute_w_dr()
    zero_shot()