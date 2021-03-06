import random
import time
import copy

import torch

from utils.functions import batchify_with_label, epistemic_uncertainty, decode_seq
from model.mc_model import MCmodel
from utils.data import Data

USE_CUDA = torch.cuda.is_available()


def load_model(data):
    print("Load Base Model from dir: ", data.model_dir)
    model = MCmodel(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name))

    return model


# Agent 使用 env 的 step 和 reset 这两个方法完成交互
class Env(object):
    def __init__(self, data=None, model_dir=None):
        print("Creat environment...")
        # 加载训练好的基础模型
        if data is not None:
            self.data = data
        elif model_dir is not None:
            self.data = Data()
            self.data.load(model_dir + "/data.dset")
        else:
            print("Error: Unable to create base model")
            exit(1)
        self.model = load_model(self.data)
        # 设置 model 为评估模式
        self.model.eval()

        self.label_alphabet = self.data.label_alphabet

        # 用来寻找同一篇文章中一个单词的其他出现位置
        self.word_mat = self.data.word_mat

        self.max_read_memory = self.data.max_read_memory
        self.threshold = self.data.threshold

        # 数据集
        self.train_Ids = self.data.train_Ids + self.data.dev_Ids
        self.test_Ids = self.data.test_Ids
        self.train_doc_total_num = len(self.train_Ids)
        self.test_doc_total_num = len(self.test_Ids)

        print("训练集文章数为 %s" % self.train_doc_total_num)
        print("测试集文章数为 %s" % self.test_doc_total_num)

        self.pred_label_total_result = [0] * (self.train_doc_total_num + self.test_doc_total_num)
        self.gold_label_total_result = [0] * (self.train_doc_total_num + self.test_doc_total_num)
        self.uncertainty_total_result = [0] * (self.train_doc_total_num + self.test_doc_total_num)
        self.lstm_out_total_result = [0] * (self.train_doc_total_num + self.test_doc_total_num)
        self.outs_total_result = [0] * (self.train_doc_total_num + self.test_doc_total_num)
        start = time.time()
        self.get_all_result()
        end = time.time()
        print("使用基模型获得所有预测结果使用时: %.2fs" % (end-start))

        # 当前处理的文档
        self.cur_doc_idx = -1
        # 当前处理文章的标号
        self.cur_doc_num = -1
        # 当前文档单词总数
        self.cur_doc_word_total_num = -1
        # 当前处理文档的基模型预测结果
        self.pred_label_result = None
        self.gold_label_result = None
        self.lstm_out_result = None
        self.outs_result = None
        self.uncertainty_result = None

        # 当前处理的单词
        self.cur_word_idx = -1
        # 当前处理单词的参考单词（同一篇文章中的不同出现）
        self.cur_word_reference = None
        self.cur_word_reference_idx = -1
        # 可以参考的单词总数
        self.cur_word_reference_num = -1

        # 测试时使用
        self.gold_results = []
        self.pred_results = []

        # 结束状态向量
        self.state_dim = None
        self.end_state = None

        self.time_step = -1

    # 环境初始化
    # 返回初始状态向量
    def reset(self, if_train=True):
        print("Environment initializing...")

        self.time_step = -1

        # if if_train:
        #     random.shuffle(self.train_Ids)

        if not if_train:
            self.gold_results = []
            self.pred_results = []

        self.cur_doc_idx = -1
        self.next_doc(if_train)

        return self.get_state()

    def step(self, action, if_train=True):
        reward = 0
        done = False
        info = None

        if -1 < action < 17:
            # print("------------------------------------------")
            # print("当前单词的正确标签是 %s" % self.gold_label_result[self.cur_word_idx])
            # print("当前单词的预测标签是 %s" % self.pred_label_result[self.cur_word_idx])
            # print("当前单词的要变成的标签是 %s" % (action + 1))

            # 当前标签变为 action+1
            self.pred_label_result[self.cur_word_idx] = action + 1
            reward += 1 if (action + 1) \
                == self.gold_label_result[self.cur_word_idx] else -1

            done = self.next_word(if_train)
        elif action == 17:
            # print("切换下一个参考单词")
            reward += 0.1
            done = self.next_reference(if_train)

        if not done:
            new_state = self.get_state()
        else:
            info = (self.label_recover())
            new_state = self.end_state

        self.time_step += 1
        if self.time_step % 1000 == 0:
            print("---------- Time Step %s ----------" % self.time_step)
            print("当前处理的文档数为", self.train_doc_total_num if if_train else self.test_doc_total_num, end=' ')
            print("当前处理第 %s 篇文档" % self.cur_doc_idx)
            print("当前文档有单词数", self.cur_doc_word_total_num, end=' ')
            print("当前处理第 %s 个单词" % self.cur_word_idx)
            print("该单词可以参考的单词数", self.cur_word_reference_num, end=' ')
            print("当前查看其第 %s 个参考" % self.cur_word_reference_idx)

        return reward, new_state, done, info

    def get_state(self):
        state = []
        state.extend(self.lstm_out_result[self.cur_word_idx])
        state.extend(self.lstm_out_result[self.cur_word_reference[self.cur_word_reference_idx] - 1])
        state.extend(self.outs_result[self.cur_word_idx])
        state.extend(self.outs_result[self.cur_word_reference[self.cur_word_reference_idx] - 1])
        state.append(self.cur_word_reference_num - self.cur_word_reference_idx - 1)
        state.append(self.pred_label_result[self.cur_word_idx])
        state.append(self.pred_label_result[self.cur_word_reference[self.cur_word_reference_idx] - 1])
        if state[-1] == state[-2]:
            state.append(1)
        else:
            state.append(0)
        state.append(self.uncertainty_result[self.cur_word_idx])
        state.append(self.uncertainty_result[self.cur_word_reference[self.cur_word_reference_idx] - 1])
        state.append(abs(state[-1] - state[-2]))

        if self.state_dim is None:
            self.state_dim = len(state)
            self.end_state = [0] * self.state_dim

        return state

    def get_all_result(self):
        for i in range(self.train_doc_total_num):
            document = self.train_Ids[i:i+1]
            doc_num = document[0][0][4]
            pred_label_result, gold_label_result, lstm_out_result, \
                outs_result, uncertainty_result = self.predict_by_base_model(self.model, document)
            self.pred_label_total_result[doc_num] = pred_label_result
            self.gold_label_total_result[doc_num] = gold_label_result
            self.uncertainty_total_result[doc_num] = uncertainty_result
            self.outs_total_result[doc_num] = outs_result
            self.lstm_out_total_result[doc_num] = lstm_out_result

        for i in range(self.test_doc_total_num):
            document = self.test_Ids[i:i+1]
            doc_num = document[0][0][4]
            pred_label_result, gold_label_result, lstm_out_result, \
                outs_result, uncertainty_result = self.predict_by_base_model(self.model, document)
            self.pred_label_total_result[doc_num] = pred_label_result
            self.gold_label_total_result[doc_num] = gold_label_result
            self.uncertainty_total_result[doc_num] = uncertainty_result
            self.outs_total_result[doc_num] = outs_result
            self.lstm_out_total_result[doc_num] = lstm_out_result

    # 使用基础模型，逐文档进行处理，并保存中间结果，供强化学习使用
    def predict_by_base_model(self, model, instance):
        with torch.no_grad():
            batch_word, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, batch_label, mask, doc_idx, word_idx \
                = batchify_with_label(instance, USE_CUDA, False)

            batch_size = mask.size(0)  # 句子数
            seq_len = mask.size(1)  # 最长句子长度

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

            for idx in range(batch_size):
                for idy in range(seq_len):
                    if mask[idx][idy] != 0:
                        pred_label_result.append(pred_label[idx][idy])
                        gold_label_result.append(gold_label[idx][idy])
                        lstm_out_result.append(lstm_out[idx][idy])
                        outs_result.append(outs[idx][idy])
                        uncertainty_result.append(uncertainty[idx][idy])

        return pred_label_result, gold_label_result, lstm_out_result, outs_result, uncertainty_result

    # 切换文章
    def next_doc(self, if_train=True):
        if (not if_train) and self.cur_word_idx != -1:
            # 统计上一篇文章的处理结果
            self.gold_results.append(self.gold_label_result)
            self.pred_results.append(self.pred_label_result)

        self.cur_doc_idx += 1
        self.cur_word_idx = -1

        if if_train:
            doc_num = self.train_doc_total_num
            doc_Ids = self.train_Ids
        else:
            doc_num = self.test_doc_total_num
            doc_Ids = self.test_Ids

        if self.cur_doc_idx == doc_num:
            # 表明处理完毕
            return True

        # 获得文章的预测结果
        document = doc_Ids[self.cur_doc_idx:self.cur_doc_idx + 1]
        # Ids: 文档数 × 句子数 × [word_Ids, char_Ids, label_Ids, word_idx, d_idx]
        self.cur_doc_num = document[0][0][4]
        self.pred_label_result = copy.deepcopy(self.pred_label_total_result[self.cur_doc_num])
        self.gold_label_result = self.gold_label_total_result[self.cur_doc_num]
        self.lstm_out_result = self.lstm_out_total_result[self.cur_doc_num]
        self.outs_result = self.outs_total_result[self.cur_doc_num]
        self.uncertainty_result = self.uncertainty_total_result[self.cur_doc_num]

        # 当前文档单词总数
        self.cur_doc_word_total_num = len(self.pred_label_result)

        # print("当前处理第 %s 篇文章，文章号码为 %s，当前文章单词数为 %s" %
        #       (self.cur_doc_idx, self.cur_doc_num, self.cur_doc_word_total_num))

        # 切换单词
        return self.next_word(if_train)

    # 切换单词
    def next_word(self, if_train=True):
        self.cur_word_idx += 1
        self.cur_word_reference_idx = -1

        if self.cur_word_idx == self.cur_doc_word_total_num:
            # print("当前文档单词处理完毕，处理下一篇文档")
            return self.next_doc(if_train)

        while if_train and self.pred_label_result[self.cur_word_idx] == self.gold_label_result[self.cur_word_idx]:
            # print("训练时，跳过预测正确的单词")
            self.cur_word_idx += 1
            self.cur_word_reference_idx = -1
            if self.cur_word_idx == self.cur_doc_word_total_num:
                # print("当前文档单词处理完毕，处理下一篇文档")
                return self.next_doc(if_train)

        while (not if_train) and self.uncertainty_result[self.cur_word_idx] < self.threshold:
            # print("测试时，跳过不确定度低的单词")
            self.cur_word_idx += 1
            self.cur_word_reference_idx = -1
            if self.cur_word_idx == self.cur_doc_word_total_num:
                # print("当前文档单词处理完毕，处理下一篇文档")
                return self.next_doc(if_train)

        self.cur_word_reference = self.word_mat[self.cur_doc_num][self.cur_word_idx + 1]
        try:
            self.cur_word_reference_num = self.cur_word_reference.tolist().index(0)
        except ValueError:
            self.cur_word_reference_num = self.max_read_memory
        # 跳过单独出现的单词
        if self.cur_word_reference_num == 0:
            # print("当前处理第 %s 个单词，其没有参考单词，故跳过" % self.cur_word_idx)
            return self.next_word(if_train)

        # print("当前处理第 %s 个单词，其有 %s 个参考单词" %
        #       (self.cur_word_idx, self.cur_word_reference_num))

        return self.next_reference(if_train)

    # 切换参考单词
    def next_reference(self, if_train=True):
        self.cur_word_reference_idx += 1
        if self.cur_word_reference_idx == self.cur_word_reference_num:
            # print("当前单词已参考完全部单词，切换下一个单词")
            return self.next_word(if_train)
        # print("当前查看其第 %s 个参考" % self.cur_word_reference_idx)

        return False

    # 将 index 切换成真实的标签
    def label_recover(self):
        new_pred_results = []
        new_gold_results = []

        for idx in range(len(self.pred_results)):  # 按文章遍历
            pred = [self.label_alphabet.get_instance(self.pred_results[idx][idy])
                    for idy in range(len(self.pred_results[idx]))]
            gold = [self.label_alphabet.get_instance(self.gold_results[idx][idy])
                    for idy in range(len(self.gold_results[idx]))]
            new_pred_results.append(pred)
            new_gold_results.append(gold)

        return new_pred_results, new_gold_results

