import random

import torch
import numpy as np

from utils.functions import batchify_with_label, epistemic_uncertainty, decode_seq
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
            self.data = Data()
            self.data.load(model_dir + "/data.dset")
            self.data.model_dir = 'outs_20210319'
        else:
            print("Error: Unable to create base model")
            exit(1)
        self.model = load_model(self.data)
        # 设置 model 为评估模式
        self.model.eval()

        self.word_mat = self.data.word_mat

        self.train_Ids = self.data.train_Ids + self.data.dev_Ids
        self.test_Ids = self.data.test_Ids

        self.train_doc_total_num = len(self.data.train_Ids) + len(self.data.dev_Ids)
        self.test_doc_total_num = len(self.data.test_Ids)

        # 当前处理的文档和单词
        self.cur_doc_idx = -1
        self.cur_word_idx = -1
        # 当前处理文章读取时候的标号
        self.cur_doc_num = -1

        # 当前处理文档的基模型预测结果
        self.pred_label_result = None
        self.gold_label_result = None
        self.lstm_out_result = None
        self.outs_result = None
        self.uncertainty_result = None

        # 当前处理单词的参考单词（同一篇文章中的不同出现）
        self.cur_word_reference = None

    # 环境初始化
    # 返回初始状态向量
    def reset(self):
        random.shuffle(self.train_Ids)

        self.cur_doc_idx = 0
        self.cur_word_idx = 1

        # 使用基模型预测当前所选文章
        document = self.train_Ids[self.cur_doc_idx:self.cur_doc_idx+1]
        self.cur_doc_num = document
        self.pred_label_result, self.gold_label_result, self.lstm_out_result, self.outs_result,\
            self.uncertainty_result = self.predict_by_base_model(self.model, document)

        self.cur_word_reference = self.word_mat[0]


    def setp(self):
        pass

    # 使用基础模型，逐文档进行处理，并保存中间结果，供强化学习使用
    def predict_by_base_model(self, model, instance):
        with torch.no_grad():
            batch_word, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, batch_label, mask, doc_idx, word_idx \
                = batchify_with_label(instance, self.data.use_gpu, True)

            mask = mask.eq(1)
            p, lstm_out, outs, word_represent = model.MC_sampling(batch_word, batch_wordlen, batch_char,
                                                                  batch_charlen,
                                                                  batch_charrecover, self.data.nsamples)
            model1_preds = decode_seq(outs, mask)
            uncertainty = epistemic_uncertainty(p, mask)

            # 恢复成正常句子顺序
            pred_label = model1_preds[batch_wordrecover].cpu().data.numpy()
            gold_label = batch_label[batch_wordrecover].cpu().data.numpy()
            lstm_out = lstm_out[batch_wordrecover].cpu().data.numpy()
            outs = outs[batch_wordrecover].cpu().data.numpy()
            uncertainty = uncertainty[batch_wordrecover].cpu().data.numpy()
            mask = mask[batch_wordrecover].cpu().data.numpy()

            # 一行表示一个单词
            pred_label_result = []
            gold_label_result = []
            lstm_out_result = []
            outs_result = []
            uncertainty_result = []

            batch_size = mask.size(0)  # 句子数
            seq_len = mask.size(1)  # 最长句子长度
            for idx in range(batch_size):
                for idy in range(seq_len):
                    if mask[idx][idy] != 0:
                        pred_label_result += pred_label[idx][idy]
                        gold_label_result += gold_label[idx][idy]
                        lstm_out_result += lstm_out[idx][idy]
                        outs_result += outs[idx][idy]
                        uncertainty_result += uncertainty[idx][idy]

        return pred_label_result, gold_label_result, lstm_out_result, outs_result, uncertainty_result


print(np.random.permutation(10))
