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
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DLONGTYPE = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# 相关超参数设定
EPOCHS = 200000
STATE_VEC_DIM = 400 + 400 + 18 + 18 + 1 + 3 + 3
DQN_HIDDEN_DIM1 = 2000
DQN_HIDDEN_DIM2 = 500
NUM_ACTIONS = 18

GAMMA = 0.9
REPLAY_SIZE = 8000
BATCH_SIZE = 256
REPLACE_TARGET_FREQ = 1000

LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )


def dqn_learn(env,
              exploration=LinearSchedule(EPOCHS // 2, 0.1),
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
        total_loss = 0
        # 环境初始化
        obs = env.reset()  # obs 是一个 list
        while True:
            # 选择动作
            sample = random.random()
            threshold = exploration.value(epoch_id)
            if sample > threshold:
                observation = torch.tensor(obs).unsqueeze(0).type(DTYPE)  # observation 是一个 tensor
                value = Q(observation).cpu().data.numpy()
                action = value.argmax(-1)[0]
            else:
                action = np.random.randint(NUM_ACTIONS)
            # 执行动作
            reward, new_obs, done, _ = env.step(action)

            replay_buffer.append((obs, action, reward, new_obs, done))
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.popleft()

            obs = new_obs

            if len(replay_buffer) > BATCH_SIZE:
                # print("执行经验回放")

                # 首先准备输入数据
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                state_batch = [data[0] for data in minibatch]
                action_batch = [data[1] for data in minibatch]
                reward_batch = [data[2] for data in minibatch]
                next_state_batch = [data[3] for data in minibatch]
                done_batch = [data[4] for data in minibatch]
                # 第一维是 batch_size
                state_tensor = torch.tensor(state_batch).type(DTYPE)
                action_tensor = torch.tensor(action_batch).type(DLONGTYPE)
                reward_tensor = torch.tensor(reward_batch).type(DTYPE)
                next_state_tensor = torch.tensor(next_state_batch).type(DTYPE)
                done_tensor = torch.tensor(done_batch).type(DTYPE)

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

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_param_updates += 1
                if num_param_updates % REPLACE_TARGET_FREQ == 0:
                    Q_target.load_state_dict(Q.state_dict())

            if done:
                break

        end = time.time()
        print("Epoch %s  Time: %.2f s  Total Loss: %.2f" % (epoch_id, end - start, total_loss))
        # 每训练 10000 步，进行一次评估
        if epoch_id % 5 == 0:
            print("--------------------------------------------------------")
            print("--------------进入测试阶段")
            print("--------------------------------------------------------")

            obs = env.reset(False)

            while True:
                observation = torch.tensor(obs).unsqueeze(0).type(DTYPE)
                value = Q(observation).cpu().data.numpy()
                action = value.argmax(-1)[0]
                reward, new_obs, done, info = env.step(action, False)
                obs = new_obs
                if done:
                    gold_results = info[0]
                    pred_results = info[1]
                    break
            acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
            print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f; \n" % (acc, p, r, f))
