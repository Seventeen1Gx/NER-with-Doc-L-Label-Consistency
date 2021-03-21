import argparse
import os
import random
import sys
import time

import gc
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from model.env import Env
from model.mc_model import MCmodel
from utils.data import Data
from utils.learn import dqn_learn
from utils.metric import get_ner_fmeasure
from utils.functions import batchify_with_label, epistemic_uncertainty, decode_seq


def data_initialization(data):
    # 从训练集、开发集、测试集创建 label_alphabet、word_alphabet、char_alphabet
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    # 关闭 alphabet 的更新
    data.fix_alphabet()


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print("Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# outs: 总句子数 × 最大句子长度，每个元素值对应 label_alphabet 中的 index
# batch_label: 跟上面相同，不过这里是真实的标签
def get_loss(outs, batch_label, weight=None):
    batch_size, seq_len = outs.size()[:2]
    # 创建交叉熵损失函数
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    # 调用（按 token 看）
    loss = loss_function(outs.view(batch_size * seq_len, -1), batch_label.view(batch_size * seq_len))
    if weight is not None:
        # 加权
        loss = (loss * weight.reshape(-1))
    return loss.sum()


# pred_variable: model 的 forward 返回
# gold_variable: tensor(总句子数，最长句子长度)
def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    # batch_size = gold_variable.size(0)  # 句子数
    seq_len = gold_variable.size(1)  # 最长句子长度
    # cuda 数据转换成 cpu
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        # get_instance 根据 index 从 alphabet 恢复出 token
        # [O S-LOC B-PER O ...]
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


# 使用 model 在 name 这个数据集上评估
def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)

    gold_results = []
    pred_results = []

    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)  # 文档数
    total_batch = train_num // batch_size + 1

    # 设置 model 为评估模式
    model.eval()

    with torch.no_grad():  # 起始时，清空梯度
        for batch_id in range(total_batch):
            # 取 instances[start:end]
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue

            # 将数据 instances[start:end] batchify
            batch_word, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, batch_label, mask, doc_idx, word_idx \
                = batchify_with_label(instance, data.use_gpu, False)

            mask = mask.eq(1)  # 转化为 T/F
            # 重复多次获得结果，再取平均
            p, lstm_out, outs, word_represent = model.MC_sampling(batch_word, batch_wordlen, batch_char, batch_charlen,
                                                                  batch_charrecover, data.nsamples)

            # 获得每个 token 的预测结果标签在 alphabet 中的 index
            model1_preds = decode_seq(outs, mask)
            # 获得每个 token 对应的不确定度
            uncertainty = epistemic_uncertainty(p, mask)

            # 总句子数 × [O O O O]
            pred_labels, gold_label = recover_label(model1_preds, batch_label, mask, data.label_alphabet,
                                                    batch_wordrecover)
            gold_results += gold_label
            pred_results += pred_labels

    decode_time = time.time() - start_time
    speed = train_num / decode_time

    # 得到测量指标
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    score = f
    print("%s: time: %.2f s, speed: %.2f doc/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f; \n" %
          (name, decode_time, speed, acc, p, r, f))

    # 保存预测结果
    if name == 'raw':
        print("save predicted results to %s" % data.decode_dir)
        data.convert_doc_to_sent(name)
        data.write_decoded_results(pred_results, name)

    return score, pred_results


def load_model_decode(data):
    print("Load Model from dir: ", data.model_dir)
    model = MCmodel(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name))

    evaluate(data, model, "raw")


def train_mc_model(data):
    print("Training base model...")
    # 打印 data 总览
    data.show_data_summary()
    save_data_name = data.model_dir + "/data.dset"
    if data.save_model:
        data.save(save_data_name)

    batch_size = data.batch_size  # 一次 batch 的文档数
    train_num = len(data.train_Ids)  # 训练集文档数
    total_batch = train_num // batch_size + 1  # 整数除法

    model = MCmodel(data)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("pytorch total params: %d" % pytorch_total_params)

    # 创建优化器
    lr_detail1 = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": data.lr},
                  ]
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(lr_detail1,
                              momentum=data.momentum, weight_decay=data.l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(lr_detail1, weight_decay=data.l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(lr_detail1, weight_decay=data.l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(lr_detail1, weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(lr_detail1, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s" % data.optimizer)
        exit(1)

    best_dev = -10  # 记录最好的 dev 测试分数
    best_test = -10  # 记录 best_dev 这个模型的 test 测试分数

    max_test = -10  # 记录出现过的最大的 test 测试分数
    max_test_epoch = -1  # 记录得到 max_test 是第几个 epoch
    max_dev_epoch = -1  # 记录得到 max_dev 是第几个 epoch

    for idx in range(data.m1_iteration):
        epoch_start = time.time()
        print("\n ###### Epoch: %s/%s ######" % (idx, data.m1_iteration))
        if data.optimizer.lower() == "sgd":
            # sgd 需要手动进行 lr_decay
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)

        # 每个 Epoch 前 loss 初始化为 0
        sample_loss = 0
        total_loss = 0

        # 每个 Epoch 打乱文档
        random.shuffle(data.train_Ids)

        # 将模型设置成训练模式（只是状态设置，没有实际动作）
        # 根据状态有不同的执行路径
        # 因为每次 epoch 结束，会在 dev 和 test 上评估，评估操作中会将模型设置成评估模式
        model.train()

        # 梯度置零，进入 batch
        model.zero_grad()
        for batch_id in range(total_batch):
            # 当前使用 [start:end) 文档进行一次梯度下降计算
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]

            if not instance:
                continue

            # batch_word: tensor(总句子数, 最长句子长度)，记录句子中单词在 alphabet 中的 index，并且长句在前
            # batch_wordlen: tensor(总句子数,)，记录句子长度
            # batch_wordrecover: 恢复句子顺序用的 indices
            # batch_char: tensor(总句子数，最长句子长度，最长单词长度)
            # batch_charlen: tensor(总句子数，最长句子长度)，记录单词长度
            # batch_charrecover: 恢复单词顺序用的 indices
            # batch_label: tensor(总句子数，最长句子长度)
            # mask: tensor(总句子数，最长句子长度)，记录上面二维 tensor 的有效位
            # 1 表示有效，就说明对应位置上是有 token 的，而不是对齐填充
            # doc_idx: tensor(总句子数)，记录句子属于哪篇文档
            # word_idx: tensor(总句子数, 最大句子长度)，记录句子单词在文档中出现的 index
            batch_word, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, batch_label, mask, doc_idx, word_idx \
                = batchify_with_label(instance, data.use_gpu, True)

            # 总句子数 × 最长句子长度
            instance_num, seq_len = batch_word.size()
            # 0/1 值换成 F/T
            mask = mask.eq(1)

            # mcmodel.forward
            # p: 总句子数 × 句子长度 × label_alphabet_size 标签分布
            # lstm_out: 总句子数 × 句子长度 × hidden_dim
            # outs: 总句子数 × 句子长度 × label_alphabet_size -- outs1 经过 softmax 得到 p
            # word_represent: 总句子数 × sent_len × input_size
            p, lstm_out, outs, word_represent = model(batch_word, batch_wordlen, batch_char, batch_char, batch_charrecover)

            # 总句子数 × 最大句子长度，每个元素值对应 label_alphabet 中的 index
            model1_preds = decode_seq(outs, mask)
            # 总句子数 × 最大句子长度，每个元素值是对应 label 的不确定度
            uncertainty = epistemic_uncertainty(p, mask)

            loss = get_loss(outs, batch_label)

            if data.average_batch_loss:
                loss = loss / instance_num

            sample_loss += loss.item()  # 采样损失，避免非法
            total_loss += loss.item()  # 一个 epoch 的全部损失，即整个训练数据集上的损失

            if end % 500 == 0:
                # 检查 sample_loss 是否合法
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0

            # 反向传播，计算好梯度值
            loss.backward()
            clip_grad_norm_(model.parameters(), data.clip_grad)

            # 梯度更新
            optimizer.step()
            # 置零，然后进入下一 batch
            model.zero_grad()

        # 一个 epoch 结束，打印统计信息
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2f s, speed: %.2f doc/s,  total loss: %s"
              % (idx, epoch_cost, train_num / epoch_cost, total_loss))

        # 损失过大，直接退出
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # 用每个 epoch 训练出来的 model 在 dev 和 test 上评估

        # dev
        dev_score, _ = evaluate(data, model, "dev")

        # test
        test_score, _ = evaluate(data, model, "test")

        if max_test < test_score:
            max_test_epoch = idx
        max_test = max(test_score, max_test)
        if dev_score > best_dev:
            print("Exceed previous best dev score")
            best_test = test_score
            best_dev = dev_score
            max_dev_epoch = idx
            if data.save_model:
                model_name = data.model_dir + "/best_model.ckpt"
                print("Save current best model in file:", model_name)
                torch.save(model.state_dict(), model_name)

        print("Score summary: max dev (%d): %.4f, test: %.4f; max test (%d): %.4f" % (
            max_dev_epoch, best_dev, best_test, max_test_epoch, max_test))

        # 垃圾回收
        gc.collect()


if __name__ == '__main__':
    # 创建用来解析命令行参数的工具
    parser = argparse.ArgumentParser(description='Tuning model')
    # 添加命令行参数
    parser.add_argument('--train_dir', default='data/conll2003/train.txt')
    parser.add_argument('--dev_dir', default='data/conll2003/dev.txt')
    parser.add_argument('--test_dir', default='data/conll2003/test.txt')
    parser.add_argument('--raw_dir', default='data/conll2003/test.txt')
    parser.add_argument('--model_dir', default='outs', help='the prefix of output directory, '
                                                            'we will creat a new directory based on the prefix and '
                                                            'a generated random number to prevent overwriting.')

    parser.add_argument('--save_model', default=True)

    parser.add_argument('--word_emb_dir', default='data/glove.6B.100d.txt')
    parser.add_argument('--norm_word_emb', default=False, help='whether to normalize word embedding')
    parser.add_argument('--number_normalized', default=True, help='whether to normalize number')

    parser.add_argument('--status', choices=['train', 'decode'], default='train')
    parser.add_argument('--m1_iteration', default=100, help='epoch of model1')
    parser.add_argument('--batch_size', default=1, help='number of documents in a batch')
    parser.add_argument('--ave_batch_loss', default=True)
    parser.add_argument('--seed', default=333)

    parser.add_argument('--use_char', default=True)
    parser.add_argument('--char_emb_dim', default=30)
    parser.add_argument('--char_seq_feature', choices=['CNN', 'CNN3', 'GRU', 'LSTM'], default='CNN')
    parser.add_argument('--char_hidden_dim', default=50)
    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--wordRep_dropout', default=0.5, help='dropout after representation layer')

    parser.add_argument('--bayesian_lstm_dropout', default=0.01, help='dropout in & between lstm layers')
    parser.add_argument('--model1_dropout', default=0.25, help='dropout in output fully connected layer')
    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--model1_layer', default=1)
    parser.add_argument('--bilstm', default=True)
    parser.add_argument('--nsample', default=32, help='sample times in testing period, we use 1 for training period')
    parser.add_argument('--threshold', default=0.1, help='the threshold to combine results of two stages, '
                                                         'a smaller one prefers results from the second stage.')

    parser.add_argument('--max_read_memory', default=10)

    parser.add_argument("--gpu_id", type=int, default=0, help="ID of GPU to be used")

    parser.add_argument('--clip_grad', default=1)
    parser.add_argument('--l2', default=1e-6)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--learning_rate', default=0.015)
    parser.add_argument('--lr_decay', default=0.05)
    parser.add_argument('--momentum', default=0.9)

    parser.add_argument('--train_base_model', type=bool, default=False, help='whether to train base model'
                                                                             'if not, load base model from model dir')
    parser.add_argument('--train_DQN', type=bool, default=True, help='whether to train DQN'
                                                                     'if not, load DQN from model dir')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机数种子
    seed_num = int(args.seed)
    print("Seed num:", seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)

    # 创建 Data 对象，这个对象中保存了代码运行中要用到的各种参数
    data = Data()
    # 根据运行环境添加是否使用GPU进行运算这一属性
    data.use_gpu = torch.cuda.is_available()
    # 设置使用的 GPU
    if data.use_gpu:
        torch.cuda.set_device(args.gpu_id)

    if args.status == 'train':
        print("MODE: train")

        if args.train_base_model:
            data.read_config(args)

            # 创建模型保存文件夹，并通过 uuid 避免覆盖
            uid = time.strftime("%Y%m%d%H%M%S", time.localtime())
            data.model_dir = data.model_dir + "_" + uid
            print("model dir: %s" % uid)
            if not os.path.exists(data.model_dir):
                os.mkdir(data.model_dir)

            # data 初始化 -- 建立字母表
            data_initialization(data)
            # 生成训练实例 -- 训练集
            data.generate_instance('train')
            # 生成开发实例 -- 开发集
            data.generate_instance('dev')
            # 生成测试实例 -- 测试集
            data.generate_instance('test')
            # 建立预训练 embedding（对应单词表），没有指定目录就会跳过，用的时候用随机生成的
            data.build_pretrain_emb()
            # 训练
            train_mc_model(data)
            print("model dir: %s" % uid)

        if args.train_DQN:
            # 强化学习训练
            if args.train_base_model:
                env = Env(data=data)
            else:
                args.model_dir = 'outs_20210320222115'  # 实际使用时，该参数从命令行获取
                env = Env(model_dir=args.model_dir)
            dqn_learn(env)

    if args.status == 'decode':
        print("MODE: decode")
        # 从文件夹中读取模型
        data.load(args.model_dir + "/data.dset")
        # 命令行参数解析结果设置 data 对象
        data.read_config(args)
        # 打印 data 总览
        data.show_data_summary()
        # 从要进行 decode 的数据生成实例
        data.generate_instance('raw')

        # 基模型开始 decode 操作
        load_model_decode(data)

# nvidia-smi -l  查看 GPU 使用情况，-l 表示每秒显示一次
# Fan：N/A 是风扇转速，从 0 到 100% 之间变动，这个速度是计算机期望的风扇转速。
# 实际情况下如果风扇堵转，可能打不到显示的转速。
# 有的设备不会返回转速，因为它不依赖风扇冷却而是通过其他外设保持低温。
# Temp：是温度，单位摄氏度。
# Perf：是性能状态，从 P0 到 P12，P0 表示最大性能，P12 表示状态最小性能。
# Pwr：是能耗。
# Persistence-M：是持续模式的状态，持续模式虽然耗能大，但是在新的 GPU 应用启动时，花费的时间更少。
# Bus-Id 是涉及 GPU 总线的东西。
# Disp.A 是 Display Active，表示 GPU 的显示是否初始化。
# Memory Usage 是显存使用率。
# GPU-Util 是浮动的 GPU 利用率。
# Compute M 是计算模式。
# 下面一张表表示的是每个进程占用的显存使用率。
