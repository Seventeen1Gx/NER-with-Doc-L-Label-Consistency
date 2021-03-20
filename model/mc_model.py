from __future__ import print_function
from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep


class MCmodel(nn.Module):
    def __init__(self, data):
        super(MCmodel, self).__init__()
        self.use_gpu = data.use_gpu
        self.use_char = data.use_char
        self.model1_fc_dropout = data.model1_dropout
        self.model1_in_dropout = data.bayesian_lstm_dropout[0]
        self.bilstm_flag = data.bilstm
        self.hidden_dim = data.hidden_dim

        self.wordrep = WordRep(data)

        # 一个单词的 wordRep 的维数
        self.input_size = self.wordrep.total_size

        # 双向的话，就减半，因为最后是拼接起来的
        if self.bilstm_flag:
            lstm_hidden = data.hidden_dim // 2
        else:
            lstm_hidden = data.hidden_dim

        # 构筑多层 LSTM
        # 如果启用双向，ModuleList 中一个元素，就是两层 LSTM
        # batch_first 表示输入第一层是 batch_size
        self.lstms = nn.ModuleList([nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag)])
        for _ in range(data.model1_layer-1):
            self.lstms.append(nn.LSTM(data.HP_hidden_dim, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag))

        # 线性层
        self.hidden2tag = nn.Linear(data.hidden_dim, data.label_alphabet_size)

        if self.use_gpu:
            self.lstms = self.lstms.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        # batch × sent_len ×
        # (word_embedding_dim+char_features_extra_dim)
        # 就是 batch × sent_len × input_size
        word_represent = self.forward_word(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                           char_seq_recover)
        return self.forward_rest(word_represent, word_seq_lengths)

    # 通过 wordRep 层，获得 word_represent -- [x:c]
    def forward_word(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover):
        # WordRep 的 forward
        word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        return word_represent

    # 通过 wordRep 之后的层
    # word_represent: 总句子数 × sent_len × input_size
    # word_seq_lengths: 总句子数 × (句子长度,)
    def forward_rest(self, word_represent, word_seq_lengths):
        # 模型状态不同，运行路线不同
        if not self.training:
            # 评估状态
            # 获得 word_seq_lengths 按句子长度倒序的结果
            ordered_lens, index = word_seq_lengths.sort(descending=True)
            # 将 word_represent 按句子长度倒叙，即长句在前
            ordered_x = word_represent[index]
        else:
            # 训练状态
            # 无需排序
            ordered_x, ordered_lens = word_represent, word_seq_lengths

        # 通过 lstms 层
        for i, lstm in enumerate(self.lstms):
            # 进入之前使用一个 dropout
            ordered_x = add_dropout(ordered_x, self.model1_in_dropout)
            # 总句子数 × sent_len × input_size
            pack_input = pack_padded_sequence(ordered_x, ordered_lens.cpu(), batch_first=True)
            pack_output, _ = lstm(pack_input)
            ordered_x, _ = pad_packed_sequence(pack_output, batch_first=True)

        if not self.training:
            # 恢复正常顺序
            recover_index = index.argsort()
            lstm_out = ordered_x[recover_index]
        else:
            lstm_out = ordered_x

        # lstm_out: 总句子数 × 句子长度 × hidden_dim

        # lstm 输出结果经过 dropout 层
        h2t_in = add_dropout(lstm_out, self.model1_fc_dropout)
        # 再通过最后一层线性层
        # 总句子数 × 句子长度 × label_alphabet_size
        outs = self.hidden2tag(h2t_in)

        # 经过 softmax 层, -1 表示最后一维用来计算
        # 总句子数 × 句子长度 × label_alphabet_size
        p = F.softmax(outs, -1)
        return p, lstm_out, outs, word_represent

    # 评估时用
    # 重复 nsample 次
    def MC_sampling(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                    char_seq_recover, mc_steps):
        # 总句子数 × 最长句子长度 × input_size
        word_represent = self.forward_word(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                           char_seq_recover)
        batch, max_seq_len = word_represent.size()[:2]
        # 第一维上重复 nsample 次，其他维重复 1 次（就是保持不变）
        # 原来 5 句话，现在 5*nsample 句话
        word_represent = word_represent.repeat([mc_steps] + [1 for _ in range(1, len(word_represent.size()))])
        word_seq_lengths = word_seq_lengths.repeat([mc_steps] + [1 for _ in range(1, len(word_seq_lengths.size()))])
        # lstm_out: 总句子数 × 句子长度 × hidden_dim
        # p: 总句子数 × 句子长度 × label_alphabet_size
        # outs: 总句子数 × 句子长度 × label_alphabet_size
        p, lstm_out, outs, word_represent = self.forward_rest(word_represent, word_seq_lengths)

        # 拆回 nsample 份 word_inputs 对应的预测结果
        # 再按第一维取平均
        p = p.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        lstm_out = lstm_out.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        outs = outs.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        word_represent = word_represent.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        return p, lstm_out, outs, word_represent

# x 通过一个 dropout 层
def add_dropout(x, dropout):
    # x: batch × sent_len × input_size
    # x.transpose(1, 2): batch × sent_len × input_size
    return F.dropout2d(x.transpose(1, 2)[..., None], p=dropout, training=True).squeeze(-1).transpose(1, 2)
