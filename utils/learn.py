import random
import time
from collections import namedtuple, deque

import torch
from torch import optim
import numpy as np

from model.dqn import DQN
from utils.metric import get_ner_fmeasure

from utils.schedule import LinearSchedule


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# 相关超参数设定
EPOCHS = 500000
STATE_VEC_DIM = 400 + 400 + 18 + 18 + 1 + 3 + 3
DQN_HIDDEN_DIM1 = 1000
DQN_HIDDEN_DIM2 = 500
NUM_ACTIONS = 19

GAMMA = 0.9
REPLAY_SIZE = 1000000
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10000

LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )


def dqn_learn(env,
              exploration=LinearSchedule(1000000, 0.1),
              optimizer_spec=optimizer):
    # 初始化
    Q = DQN(STATE_VEC_DIM, DQN_HIDDEN_DIM1, DQN_HIDDEN_DIM2, NUM_ACTIONS)
    Q_target = DQN(STATE_VEC_DIM, DQN_HIDDEN_DIM1, DQN_HIDDEN_DIM2, NUM_ACTIONS)
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    replay_buffer = deque()

    loss_func = torch.nn.MSELoss()

    num_param_updates = 0

    for epoch_id in range(EPOCHS):
        print("\n ###### Epoch: %s/%s ######" % (epoch_id, EPOCHS))
        start = time.time()
        # 环境初始化
        observation = env.reset()
        while True:
            # 选择动作
            sample = random.random()
            threshold = exploration.value(epoch_id)
            if sample > threshold:
                observation = torch.tensor(observation).unsqueeze(0).type(dtype)
                value = Q(observation).cpu().data.numpy()
                action = value.argmax(-1)[0]
            else:
                action = np.random.randint(NUM_ACTIONS)
            # 执行动作
            print("------------------------------")
            print("当前执行的动作是 %s" % action)
            reward, new_obs, done, _ = env.step(action)
            if not done:
                print("获得的奖励是 %s" % reward)
            else:
                print("已处理完所有文档")

            replay_buffer.append((observation, action, reward, new_obs, done))
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.popleft()

            observation = new_obs

            if done:
                break

            if len(replay_buffer) > BATCH_SIZE:
                print("执行经验回放")

                # 首先准备输入数据
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                state_batch = [data[0] for data in minibatch]
                action_batch = [data[1] for data in minibatch]
                reward_batch = [data[2] for data in minibatch]
                next_state_batch = [data[3] for data in minibatch]
                done_batch = [data[4] for data in minibatch]
                # 第一维是 batch_size
                state_tensor = torch.tensor(state_batch).type(dtype)
                action_tensor = torch.tensor(action_batch).type(dlongtype)
                reward_tensor = torch.tensor(reward_batch).type(dtype)
                # 归一化
                # reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-7)
                next_state_tensor = torch.tensor(next_state_batch).type(dtype)
                done_tensor = torch.tensor(done_batch).type(dtype)

                # Q 网络得到的估计值
                q_values = Q(state_tensor)
                # action_tensor 的 shape 是 [32]
                # action_tensor.unsqueeze(1) 是 [32, 1]
                # q_values 是 [32, 19]
                # gather 就是取出 action_tensor 对应的动作的 Q 值
                q_s_a = q_values.gather(1, action_tensor.unsqueeze(1))
                # q_s_a 变成 [32]
                q_s_a = q_s_a.squeeze()

                # 目标值
                # .max(1) 按第 2 个维度求最大值，返回最大值以及索引
                # 所以用 [0] 取最大动作值
                # Q_target(next_state_tensor).max(1)[0]: batch_size
                target_v = reward_tensor + GAMMA * (1 - done_tensor) * Q_target(next_state_tensor).detach().max(1)[0]

                loss = loss_func(q_s_a, target_v)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                num_param_updates += 1
                if num_param_updates % REPLACE_TARGET_FREQ == 0:
                    Q_target.load_state_dict(Q.state_dict())
        end = time.time()
        print("Epoch %s  Time: %.2f s" % (epoch_id, end - start))
        # 每训练 10000 步，进行一次评估
        if epoch_id % 10000:
            observation = env.reset(False)

            while True:
                observation = torch.tensor(observation).unsqueeze(0).type(dtype)
                value = Q(observation).cpu().data.numpy()
                action = value.argmax(-1)[0]
                reward, new_obs, done, info = env.step(action, False)
                observation = new_obs
                if done:
                    gold_results = info[0]
                    pred_results = info[1]
                    break
            acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
            print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f; \n" % (acc, p, r, f))
