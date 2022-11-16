from envs.dual_arm_env_continuous import DualArmContinuousEnv
from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
import yaml
import os
from params import scale_reward

# do not render the scene
e_render = False

# 载入环境参数
with open(os.path.join(os.path.dirname(__file__), "config/dualarmcontinuous.yaml"), "r") as f:
    env_args = yaml.load(f, yaml.FullLoader)
world = DualArmContinuousEnv(**env_args)

vis = visdom.Visdom(port=8097)
reward_record = []

np.random.seed(env_args["seed"])
th.manual_seed(env_args["seed"])

n_agents = 2
n_states = 95
n_obs = 13
n_actions = 3
capacity = 1000000 # replay memory capacity, 超过容量后新数据覆盖旧数据
batch_size = 1000  # train 的时候采样的 transition 的数量

n_steps = 100000000  # 总训练步数
max_steps = 100  # 每 episode 最大步长
steps_before_train = 1200  # 稍微大于 batch_size

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_obs, n_actions, batch_size, capacity,
                steps_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
while maddpg.steps_done < n_steps:
    state, obs = world.reset()
    state = th.from_numpy(state).float()
    obs = np.stack(obs)
    obs = th.from_numpy(obs).float()

    total_reward = 0.0
    rr = np.zeros((n_agents,))
    last_act = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    last_act = th.from_numpy(last_act).float()
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        if maddpg.steps_done % 100 == 0 and e_render:
            world.render()

        obs = obs.type(FloatTensor)
        last_act = last_act.type(FloatTensor)
        input = th.cat([obs, last_act], dim=1)

        action = maddpg.select_action(input).data.cpu()
        state_, obs_, reward, done, _ = world.step(action.numpy())

        state_ = th.from_numpy(state_).float()
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        reward = np.stack(reward)
        reward = th.from_numpy(reward).float()

        if t != max_steps - 1 or not done:
            next_state = state_
            next_obs = obs_
        else:
            next_state = None
            next_obs = None

        total_reward += reward.sum()
        rr += reward.numpy()

        maddpg.memory.push(state, obs.data, action, next_state, next_obs, reward, last_act.data)
        obs = next_obs
        last_act = action

        c_loss, a_loss = maddpg.update_policy()

        if done:
            break

    print('Step: %d, reward = %f' % (maddpg.steps_done, total_reward))
    reward_record.append(total_reward)

    if maddpg.steps_done == maddpg.steps_before_train:
        print('training now begins...')

    if win is None:
        win = vis.line(X=np.arange(maddpg.steps_done, maddpg.steps_done+1),
                       Y=np.array([np.append(total_reward, rr)]),
                       opts=dict(ylabel='Reward',
                                 xlabel='Episode',
                                 title='MADDPG on DualArmContinuous\n',
                                 legend=['Total'] + ['Agent-%d' % i for i in range(n_agents)]))
    else:
        vis.line(X=np.array([np.array(maddpg.steps_done).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward, rr)]),
                 win=win,
                 update='append')

    if param is None:
        param = vis.line(X=np.arange(maddpg.steps_done, maddpg.steps_done+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(ylabel='Var',
                                   xlabel='Episode',
                                   title='MADDPG on DualArmContinuous: Exploration',
                                   legend=['Variance']))
    else:
        vis.line(X=np.array([maddpg.steps_done]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')

world.close()
