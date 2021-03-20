from collections import namedtuple

from torch import optim

from model.dqn import DQN

# 相关超参数设定
EPOCHS = 500000
STATE_DIM = 100  # 两个 token 的 h 和 l 进行拼接
DQN_HIDDEN_DIM = 20
NUM_ACTIONS = 3

GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10

LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )


def dqn_learn(env, optimizer_spec=optimizer):
    # 用训练好的模型 1 来对训练集+开发集进行结果预测

    # 初始化
    Q = DQN(STATE_DIM, DQN_HIDDEN_DIM, NUM_ACTIONS)
    Q_target = DQN(STATE_DIM, DQN_HIDDEN_DIM, NUM_ACTIONS)
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    for epoch_id in range(EPOCHS):
        pass


