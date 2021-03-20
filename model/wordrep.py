from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from .charbilstm import CharBiLSTM
from .charbigru import CharBiGRU


# WordRep
class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.use_gpu = data.use_gpu  # 是否使用 GPU 进行计算
        self.use_char = data.use_char  # 是否使用 char_embedding

        self.total_size = 0

        # 声明与 char_embedding 有关的结构
        if self.use_char:
            self.extra_char_feature = False
            self.char_hidden_dim = data.char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_feature_extractor == "CNN":
                from .charcnn import CharCNN
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding,
                                            self.char_embedding_dim, self.char_hidden_dim, data.wordRep_dropout,
                                            self.use_gpu)
            elif data.char_feature_extractor == "CNN3":
                from .charcnn_3k import CharCNN
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding,
                                            self.char_embedding_dim, self.char_hidden_dim, data.wordRep_dropout,
                                            self.use_gpu)
            elif data.char_feature_extractor == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding,
                                               self.char_embedding_dim, self.char_hidden_dim, data.wordRep_dropout,
                                               self.use_gpu)
            elif data.char_feature_extractor == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), data.pretrain_char_embedding,
                                              self.char_embedding_dim, self.char_hidden_dim, data.wordRep_dropout,
                                              self.use_gpu)
            else:
                print(
                    "Error char feature selection, please check parameter data.char_feature_extractor (CNN/LSTM/GRU).")
                exit(0)

            self.total_size += self.char_feature.char_out_size

        # 声明 word_embedding 层
        self.embedding_dim = data.word_emb_dim
        # Embedding 层就是一个 W 矩阵
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        # 如果有预训练 embedding，就用上，将 W 替换成 pretrain_word_embedding
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
            self.word_embedding.weight.data = self.word_embedding.weight.data / self.word_embedding.weight.data.std()
        else:  # 没有预训练 embedding，就随机生成
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        self.total_size += self.embedding_dim

        # dropout
        self.drop = nn.Dropout(data.wordRep_dropout)

        # 有条件就用 cuda
        if self.use_gpu:
            self.word_embedding = self.word_embedding.cuda()

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        batch_size = word_inputs.size(0)  # 总句子数
        sent_len = word_inputs.size(1)  # 最大句子长度

        # input 通过 word_embedding 层，每个 word 转化成其对应的 embedding
        word_embs = self.word_embedding(word_inputs)
        # [batch × sent_len × word_embedding_dim]
        word_list = [word_embs]

        if self.use_char:
            # char_feature 是 CharCNN 对象
            # char_inputs: (batch_size*sent_len, word_length)
            # char_features: (batch_size*sent_len, char_embedding_dim) -- 单词的 char_embedding 进行了融合
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            # 恢复单词顺序，但句子顺序还是降序的
            char_features = char_features[char_seq_recover]
            # 恢复 (batch_size, sent_len, char_embedding_dim)
            char_features = char_features.view(batch_size, sent_len, -1)
            # [batch × sent_len × word_embedding_dim, batch × sent_len × feature1_embedding_dim, ...
            # , batch × sent_len × char_features_dim]
            word_list.append(char_features)

            # 在第 2 维度上接起来，就是论文中的向量 concatenate 操作，每个单词的表示更丰富
            # batch × sent_len ×
            # (word_embedding_dim+char_features_extra_dim)
            word_embs = torch.cat(word_list, 2)

        # dropout 处理
        word_embs = self.drop(word_embs)

        return word_embs


def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb
