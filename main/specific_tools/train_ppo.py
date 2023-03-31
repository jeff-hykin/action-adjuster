#%matplotlib inline
#%matplotlib widget
import torch
import gym
import torch
import gym
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from envs.warthog import WarthogEnv
import scipy.signal
import time
from config import config, path_to

# torch.manual_seed(100)
eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [
            x0,
            x1,
            x2
        ]
    output:
        [
            x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2
        ]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class ValueNetwork(nn.Module):
    def __init__(self, obs_dimension, sizes, act=nn.ReLU):
        super(ValueNetwork, self).__init__()
        sizes = [obs_dimension] + sizes + [1]
        out_activation = nn.Identity
        self.layers = []
        for j in range(0, len(sizes) - 1):
            act_l = act if j < len(sizes) - 2 else out_activation
            self.layers += [nn.Linear(sizes[j], sizes[j + 1]), act_l()]
        self.v = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.v(x)


class PolicyNetworkCat(nn.Module):
    def __init__(self, obs_dimension, sizes, action_dimension, act=nn.ReLU):
        super(PolicyNetworkCat, self).__init__()
        sizes = [obs_dimension] + sizes + [action_dimension]
        out_activation = nn.Identity
        self.layers = []
        for j in range(0, len(sizes) - 1):
            act_l = act if j < len(sizes) - 2 else out_activation
            self.layers += [nn.Linear(sizes[j], sizes[j + 1]), act_l()]
        self.pi = nn.Sequential(*self.layers)

    def forward(self, x):
        score = self.pi(x)
        # probs = F.softmax(score,dim = 1)
        dist = torch.distributions.Categorical(logits=score)
        return dist


class PolicyNetworkGauss(nn.Module):
    def __init__(self, obs_dimension, sizes, action_dimension, act=nn.ReLU):
        super(PolicyNetworkGauss, self).__init__()
        sizes = [obs_dimension] + sizes + [action_dimension]
        out_activation = nn.Identity
        self.layers = []
        for j in range(0, len(sizes) - 1):
            act_l = act if j < len(sizes) - 2 else out_activation
            self.layers += [nn.Linear(sizes[j], sizes[j + 1]), act_l()]
        self.mu = nn.Sequential(*self.layers)
        log_std = -0.5 * np.ones(action_dimension, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)

    def forward(self, x):
        mean = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        #self.act_buf = np.zeros((size,), dtype=np.float32)
        self.act_buf = np.zeros((size,act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
       # print(rews)
       # print(vals)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std()+eps)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def ppo(env, seed, buff_size, train_time_steps, gamma, clip_ratio, lr_pi, lr_vf, pi_train_itrs, v_train_itrs, lam, max_ep_len):
    obs_dim    = env.observation_space.shape
    action_dim = env.action_space.shape
    # action_dim = 2
    vi = ValueNetwork(*obs_dim, config.training.value_network.layer_sizes, act=nn.Tanh).to(device)
    pi = PolicyNetworkGauss(*obs_dim, config.training.value_network.layer_sizes, *action_dim, act=nn.Tanh).to(device)
    data_buff  = PPOBuffer(*obs_dim, *action_dim, buff_size)
    policy_opt = optim.Adam(pi.parameters(), lr=lr_pi)
    value_opt  = optim.Adam(vi.parameters(), lr=lr_vf)
    obs            = env.reset()
    curr_time_step = 0
    num_episode    = 0
    pbar           = tqdm(total=train_time_steps)
    ep_rewards     = [0]
    ep_steps       = [0]
    start_time = time.time()
    while curr_time_step < train_time_steps:
        for t in range(0, buff_size):
            curr_time_step += 1
            with torch.no_grad():
                m = pi(torch.as_tensor(obs, dtype=torch.float32).to(device))
                action = m.sample()
                # print(action)
                action = action.reshape((-1,) + action_dim)
                # print(action)
                logp = m.log_prob(action).sum(dim=1)
                # print(logp)
                action = action.cpu().numpy()
                clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
                # print(clipped_action)
                # obs_new, rew, done, _ = env.step(a.item())
                obs_new, rew, done, _ = env.step(clipped_action[0])
                ep_rewards[num_episode] += rew
                ep_steps[num_episode] += 1
                v = vi(torch.as_tensor(obs, dtype=torch.float32).to(device))
            print(f'''rew = {rew}''')
            data_buff.store(obs, action, rew, v.cpu().numpy(), logp.cpu().numpy())
            obs = obs_new
            if done or t == buff_size - 1:
                if done:
                    v_ = 0.0
                    obs = env.reset()
                    done = False
                    num_episode += 1
                    ep_rewards.append(0)
                    ep_steps.append(0)
                    curr_time = time.time()
                    if num_episode % 100 == 0:
                        print(f'episode: {num_episode-1} \t episode_reward: {np.mean(ep_rewards[-10:-2])} \t total steps:{curr_time_step}\t fps"{curr_time_step/(curr_time-start_time)}\t avg_ep_steps: {np.mean(ep_steps[-10:-2])}')
                else:
                    v_ = vi(torch.as_tensor(obs, dtype=torch.float32).to(device))
                    v_ = v_.detach().cpu().numpy()
                print(f'''v_ = {v_}''')
                data_buff.finish_path(v_)
            if curr_time_step % 100000 == 0:
                torch.save(pi, f"{path_to.temp_policy_folder}/manaul_ppo_{curr_time_step}.pt")
                np.savetxt(f"{path_to.temp_policy_folder}/avg_rew_{curr_time_step}", ep_rewards, fmt="%f")
            # curr_time_step+=1
        
        #  pbar.update(1)
        data = data_buff.get()
        print(f'''data = {data}''')
        ret      = data["ret"].to(device)
        act      = data["act"].to(device)
        adv      = data["adv"].to(device)
        o        = data["obs"].to(device)
        logp_old = data["logp"].to(device)
        
        for j in range(0, pi_train_itrs):
            policy_opt.zero_grad()
            act_dist = pi(o)
            logp = act_dist.log_prob(act).sum(dim=1)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            loss_pi.backward()
            policy_opt.step()
        for i in range(0, v_train_itrs):
            value_opt.zero_grad()
            # ret, ob = data['ret'], data['obs']
            val = vi(o)
            value_loss = F.mse_loss(val.flatten(), ret)
            value_loss.backward()
            value_opt.step()
        # pbar.update(1)
    pbar.close()
    return pi


# python.dataScience.textOutputLimit = 0

if __name__ == '__main__':
    env = WarthogEnv(path_to.waypoints_folder + "/sim_remote_waypoint.csv", None)
    pi = ppo(env, **config.training.parameters)
    # pi = torch.load("./temp_warthog.pt")
    # plt.ion()
    # plt.pause(2)
    obs = env.reset()
    for i in range(0, config.training.evaluation.timesteps):
        m = pi(torch.as_tensor(obs, dtype=torch.float32).to(device))
        action = m.sample()
        obs, rew, done, info = env.step(action.cpu().numpy())
        # env.render()
        if done:
            obs = env.reset()

    # torch.save(pi,"temp_warthog.pt")
