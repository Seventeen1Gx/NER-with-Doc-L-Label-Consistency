from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
import torch


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length,
                  word_mat, doc_idx, max_read_memory, char_padding_size=-1, char_padding_symbol='</pad>'):
    in_lines = open(input_file, 'r', encoding="utf8").readlines()

    # 语料库级别
    # 一个 instance 是一个句子
    instance_texts = []  # [[[words. chars, labels]]]
    instance_Ids = []  # [[[word_Ids. chars_Ids, labels_Ids, word_idx, d_idx]]]
    d_idx = doc_idx  # 当前处理第几篇文档，doc_idx 是传进来的文档起始下标

    # 句子级别信息，逐句更新
    words = []  # 元素是一个个 word
    word_Ids = []  # 元素是 words 中对应位置 word 在 word_alphabet 中的 idx
    labels = []  # 元素是一个个 label
    label_Ids = []  # 元素是 labels 中对应位置 label 在 label_alphabet 中的 idx
    chars = []  # 元素是一个个 list，一个 list 对应一个单词，list 中一个个元素是一个个 char
    char_Ids = []  # 元素是一个个 list，一个 list 对应一个单词，list 中一个个元素是对应 char 在 char_alphabet 中的 idx
    word_idx = []  # 一句话中的 word 处于文档中的下标

    # 文档级别，逐文档更新
    lower_word2refidx = {}  # 文档级别，refidx 是一个 list，list 是 key 在文档中出现的 w_idx（可以有多次出现）
    wix2lower_word = {}  # 文档级别，wix 是 w_idx
    w_idx = 1  # 是当前处理文档的第几个单词

    # 逐行处理
    for line in in_lines:
        if len(line) > 2:  # 有效行
            pairs = line.strip().split()

            if sys.version_info[0] < 3:
                word = pairs[0].decode('utf-8')
            else:
                word = pairs[0]

            if number_normalized:
                word = normalize_word(word)

            # 当前行是文档开始，汇总上一篇文档的信息，形成 word_mat
            if word == "-DOCSTART-":
                if wix2lower_word != {}:  # 记录了一片文档中 w_idx 和 lower_word 对应的字典
                    # Memory 对每个 word，考虑其在文档中的前 max_read_memory 次其他出现
                    # 文档单词数 × max_read_memory
                    tmp = np.zeros((w_idx, max_read_memory), dtype=np.long)
                    # 遍历文档单词（字典并不保证有序处理）
                    for _w_idx in wix2lower_word.keys():
                        # 当前处理单词在文档中出现的下标列表（从前往后的）
                        a = np.array(lower_word2refidx[wix2lower_word[_w_idx]])
                        # 将 a 中 _w_idx 元素删除，然后取前 max_read_memory 个元素
                        # 即当前处理单词其他地方出现的下标列表，并且只考虑前 max_read_memory 个
                        a = a[a != _w_idx][:max_read_memory]
                        # tmp 第 _w_idx 行前 a.size 个元素替换成 a
                        tmp[_w_idx][:a.size] = a
                    # word_mat 是语料库级别的，每篇文档对应一个 tmp
                    word_mat.append([])
                    word_mat[d_idx] = tmp
                    # 重置语料库级别的变量
                    lower_word2refidx = {}
                    wix2lower_word = {}
                    d_idx += 1
                    w_idx = 1
                continue  # 不处理 -DOCSTART- 行

            label = pairs[-1]
            labels.append(label)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            # get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:  # char 填充
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                # not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:  # 遇到空行，说明是一句话的结束，开始汇总这个句子的信息
            # max_sent_length < 0 说明没有句子长度限制
            # 忽略超过最大长度的句子
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                # 构造 wix2lower_word 和 lower_word2refidx
                for word in words:
                    key = word.lower()
                    wix2lower_word[w_idx] = key
                    lower_word2refidx.setdefault(key, [])  # 添加键，键不存在时，其值设为默认值 []，键存在则什么都不做
                    lower_word2refidx[key].append(w_idx)  # 某个单词的出现下标是按从前往后顺序的

                    word_idx.append(w_idx)
                    w_idx += 1

                # 构造 instance_texts 和 instance_Ids
                if len(instance_texts) == d_idx-doc_idx:
                    # 先按文档划分
                    instance_texts.append([])
                    instance_Ids.append([])
                # 文档内的句子划分
                instance_texts[d_idx - doc_idx].append([words, chars, labels])
                instance_Ids[d_idx - doc_idx].append([word_Ids, char_Ids, label_Ids, word_idx, d_idx])

            # 一句话统计结束，句子级别的变量置空
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
            word_idx = []

    # 最后一篇文档处理（因为它后面不是空行，是文件结束）
    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
        for word in words:
            key = word.lower()
            wix2lower_word[w_idx] = key
            lower_word2refidx.setdefault(key, [])
            lower_word2refidx[key].append(w_idx)

            word_idx.append(w_idx)
            w_idx += 1
        if len(instance_texts) == d_idx - doc_idx:
            instance_texts.append([])
            instance_Ids.append([])
        instance_texts[d_idx - doc_idx].append([words, chars, labels])
        instance_Ids[d_idx - doc_idx].append([word_Ids, char_Ids, label_Ids, word_idx, d_idx])

    if wix2lower_word != {}:
        tmp = np.zeros((w_idx, max_read_memory), dtype=np.long)
        for _w_idx in wix2lower_word.keys():
            a = np.array(lower_word2refidx[wix2lower_word[_w_idx]])
            a = a[a != _w_idx][:max_read_memory]
            tmp[_w_idx][:a.size] = a
        word_mat.append([])
        word_mat[d_idx] = tmp
        d_idx += 1

    assert d_idx-doc_idx == len(instance_texts), print("error when loading text, file format dismatch!")

    return instance_texts, instance_Ids, d_idx, word_mat  # d_idx 是当前处理好的文档数


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path is not None:
        # 将 embedding 文件内容转换成一个 dict → {token:np([1, embedd_dim])}
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    # token_num × embedding_dim
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    # 统计 alphabet 和 embedding 之间的匹配情况
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            # oov 的 embedding 用 ±scale 之间的均匀分布生成
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)  # token 数
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # embedding 文件每行是一个 token 和其对应的 embedding
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


# input_batch_list: data.train_Ids[start:end]，一个 batch 使用 [start:end) 这些文档
# gpu: 是否使用 gpu
# if_train: 是否用来训练
# 经过此方法，输入就变成了句子级别的，并且长句在前
def batchify_with_label(input_batch_list, gpu, if_train=False):
    # 多维列表根据项目拆分成多个多维列表
    # input_batch_list: 文档数 × 句子数 × [word_Ids, char_Ids, label_Ids, word_idx, d_idx]

    # 文档数 × 句子数 × [word_Ids]
    words = [[sent[0] for sent in doc] for doc in input_batch_list]
    # 文档数 × 句子数 × [char_Ids]
    chars = [[sent[1] for sent in doc] for doc in input_batch_list]
    # 文档数 × 句子数 × [label_Ids]
    labels = [[sent[2] for sent in doc] for doc in input_batch_list]
    # 文档数 × 句子数 × [word_idx]
    word_idx = [[sent[3] for sent in doc] for doc in input_batch_list]
    # 文档数 × 句子数 × (doc_idx,)
    doc_idx = [[sent[4] for sent in doc] for doc in input_batch_list]
    # map: 以 w 中的每一个元素调用 len 函数，返回包含每次 len 函数返回值的新列表
    # w: 句子数 × [100, 999, 500, 400, 500]
    # list(map(len, w)): [222, 400] -- 这篇文档两句话，长度分别是 222 和 400
    # seq_lengths: 文档数 × 句子数
    seq_lengths = [list(map(len, w)) for w in words]

    # [len(w) for w in words]: 元素个数为文档数，每个元素值是文档的句子数
    # batch_size: [start:end) 这些文档的总句子数
    batch_size = sum([len(w) for w in words])
    # max(文档数 ×(每篇文档最长句子的长度,)): 当前 batch 中最长句子的长度
    max_seq_len = max([max(list(map(len, w))) for w in words])

    # 首先声明 tensor 的大小
    # 然后填充 tensor 的值

    # 总句子数 × 最长句子长度，记录句子中单词在 alphabet 中的 index
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    # 总句子数，记录每个句子的长度
    word_seq_lengths = torch.zeros((batch_size,), requires_grad=if_train).long()
    # 总句子数 × 最长句子长度，记录句子中单词的 label 在 alphabet 中的 index
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    # 总句子数，记录每个句子属于哪个文档（从 0 开始）
    doc_idx_tensor = torch.zeros((batch_size,), requires_grad=if_train).long()
    # 总句子数 × 最长句子长度，记录句子中单词在文档中的 index（从 1 开始）
    word_idx_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    # 总句子数 × 最长句子长度（用来判断 tonsor 中哪里是有效值，哪里是无效值，因为句子长短不一致）
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()

    # 句子计数
    idx = 0
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    # zip(words, labels, seq_lengths, doc_idx, word_idx) 返回列表中的一个元素如下
    # (words 的一个元素，labels 的一个元素，seq_lengths 的一个元素，doc_idx 的一个元素，word_idx 的一个元素)
    # 句子数 × [word_Ids]
    # 句子数 × [label_Ids]
    # 句子数 × (seq_len,)
    # 句子数 × (doc_idx,)
    # 句子数 × [word_idx]
    # 第一个循环遍历文档
    for doc_seq, doc_label, doc_seqlen, doc_i, word_i in zip(words, labels, seq_lengths, doc_idx, word_idx):
        # 第二个循环遍历文档中的句子
        # [word_Ids]
        # [label_Ids]
        # (seq_len,)
        # (doc_idx,)
        # [word_idx]
        for seq, label, seqlen, dix, wix in zip(doc_seq, doc_label, doc_seqlen, doc_i, word_i):
            word_seq_lengths[idx] = seqlen
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            doc_idx_tensor[idx] = dix
            word_idx_tensor[idx, :seqlen] = torch.LongTensor(wix)
            idx += 1
    # sort by len
    # 0 表示按第 0 维排，返回排序后的结果和排序后的结果在之前列表的下标，是一个排列
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    doc_idx_tensor = doc_idx_tensor[word_perm_idx]
    word_idx_tensor = word_idx_tensor[word_perm_idx]

    # deal with char
    flatten_chars = []  # [char_Ids, ...] 元素个数为总句子数
    for doc in chars:
        for sent in doc:
            flatten_chars.append(sent)
    # char_Ids: [[[i]],[[l][o][v][e]],[[y][o][u]],[[!]]]
    # pad_chars (batch_size, max_seq_len)
    # 总句子数 × [[[i]], [[l][o][v][e]], [[y][o][u]], [[!]], [[0]], ...]
    pad_chars = [flatten_chars[idx] + [[0]] * (max_seq_len - len(flatten_chars[idx]))
                 for idx in range(len(flatten_chars))]
    # 总句子数 × [1, 4, 3, 1, ...]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    # 最长单词的字符数
    max_word_len = max(map(max, length_list))
    # 总句子数 × 最大句子长度 × 最大单词长度
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    # 总句子数 × 最大句子长度 × (字符数,)
    char_seq_lengths = torch.LongTensor(length_list)
    # 第一层循环遍历句子
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        # seq: [[[i]], [[l][o][v][e]], [[y][o][u]], [[0]], ...]
        # seq_len: [1, 4, 3, 1, ...]
        # 第二层循环遍历单词
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # word: [[l][o][v][e]]
            # wordlen: 4
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    # 按句子长度降序后
    # 转化成 (最大句子长度 × 单词数) × 最大单词长度
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    # 按单词长度降序
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]

    # 恢复原来的句子顺序要用的 indices
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    # 所有 tensor 转成 cuda 计算
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
        doc_idx_tensor = doc_idx_tensor.cuda()
        word_idx_tensor = word_idx_tensor.cuda()
    # 返回一群 tensor 和恢复顺序的 indices
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
           char_seq_recover, label_seq_tensor, mask, doc_idx_tensor, word_idx_tensor


# 计算不确定度
def epistemic_uncertainty(p, mask):
    """
    calculate epistemic_uncertainty
    :param p: (batch, max_seq_len, num_labels)
    :param mask: (batch,max_seq_len) 1 means the position is valid (not masked).
    :return:  (batch, max_seq_len)
    """
    # (batch, max_seq_len)
    # -Σplogp
    hp = -((p + 1e-30) * (p + 1e-30).log()).sum(-1)
    # 无效位置 0
    hp = hp.masked_fill(mask == 0, 0)
    return hp


# outs: 总句子数 × 句子长度 × label_alphabet_size -- 标签分布
def decode_seq(outs, mask):
    # 最后一维，替换为这一维最大值的 index
    # 即对应 label_alphabet 中的下标，就是对应 token 的 label
    preds = outs.argmax(-1)  # 总句子数 × 句子长度
    # masked_fill(mask, value): 用 value 填充 tensor 中与 mask 中值为 T 位置相对应的元素。
    # 因为 preds 得到的结果中，无效位可能不是 0 了，现在要重新置 0
    preds = preds.masked_fill(~mask, 0)  # mask padding words
    return preds


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
