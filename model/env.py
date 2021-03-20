import numpy as np
import torch

from model.mc_model import MCmodel


def load_model(data):
    print("Load Model from dir: ", data.model_dir)
    model = MCmodel(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name))

    return model


class Env(object):
    def __init__(self, data):
        print("Creat environment...")

        self.model = load_model(data)

    # 返回初始状态
    def reset(self):
        return np.random.randint(1)


class
