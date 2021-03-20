import numpy as np
import torch

from model.mc_model import MCmodel
from utils.data import Data


def load_model(data):
    print("Load Base Model from dir: ", data.model_dir)
    model = MCmodel(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name))

    return model


class Env(object):
    def __init__(self, data=None, model_dir=None, if_train=True):
        print("Creat environment...")
        self.if_train = if_train

        # 加载训练好的基础模型
        if data is not None:
            self.data = data
        elif model_dir is not None:
            self.data = Data().load(model_dir + "/data.dset")
        else:
            print("Error: Unable to create base model")
            exit(1)
        self.model = load_model(self.data)

        # 当前训练的统计信息
        # 使用 conll2003 的训练集和开发集进行强化学习 Agent 的训练
        self.doc_total_num = len(data.train_Ids) + len(data.dev_Ids)


    # 返回初始状态
    def reset(self):
        return np.random.randint(1)
