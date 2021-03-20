# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:21:40
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)  # char 经过这层得到对应的 embedding
        if pretrain_char_embedding is not None:
            # 有就用，没有就随机（这里说的是生成 char_emb 的 W 初始随机，训练中可以学出来）
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        # 窗口大小为 3（一次看 3 个 char）
        # 通过卷积，将一个单词中 char 的 embedding 汇总成一个 hidden_dim 维的隐藏向量，作为该单词的 char_rep
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.char_out_size = hidden_dim
        # 转换成 cuda
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn = self.char_cnn.cuda()

    # 通过均匀分布，为每个 token 生成一个 embedding_dim 维的 embed
    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(bsz*seq_len, word_length) -- 单词数 × 最长单词长度
                seq_lengths: numpy array (bsz*seq_len,  1)
            output:
                Variable(bsz*seq_len, char_hidden_dim) -- 一个单词对应一个 char_embedding
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        # 经过 embedding 层
        # input 每个元素变成 char_dim 维向量
        char_embeds = self.char_drop(self.char_embeddings(input))  # bsz*seq_len, word_len, char_dim
        char_embeds = char_embeds.transpose(2, 1).contiguous()  # bsz*seq_len, char_dim, word_len
        char_cnn_out = self.char_cnn(char_embeds)
        # 池化处理
        # abc 经过 embedding 层，对应 3 个 embedding_dim 维的向量
        # max_pool1d 取 3 个 embedding 中每维中的最大值，形成一个 embedding_dim 维的向量
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim) -- 一个单词多个字符对应多个 char_embedding
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)  # 单词数
        # 经过 embedding 层，再经过 dropout 处理
        # Variable(batch_size, word_length, char_hidden_dim)
        char_embeds = self.char_drop(self.char_embeddings(input))
        # 转置处理（维数索引 0 1 2，转置成 0 2 1）
        # 再连续化(语义相邻的元素，在内存中也移动成相邻，提升运算速度)
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        # 经过卷积层，再转置回来
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)
