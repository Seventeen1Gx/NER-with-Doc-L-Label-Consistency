from __future__ import print_function
from __future__ import absolute_import
from .alphabet import Alphabet
from .functions import *

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True  # 是否将文中出现的阿拉伯数字都设置成 0
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.tagScheme = "NoSeg"  # 标注策略有 BMES/BIO 两种

        # I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        # 下面三个目录的顶级目录
        self.model_dir = None
        # 预测结果文件目录
        self.decode_dir = None
        # data 对象序列化保存文件目录
        self.dset_dir = None
        # 模型文件目录
        self.load_model_dir = None

        # 传入了目录就用，没有就随机初始化
        self.word_emb_dir = None
        self.char_emb_dir = None

        # 从相应数据集 generate_instance 后得到的信息 -- 多维列表形式
        # text: 文档数 × 句子数 × [words, chars, labels]
        # words: ['I', 'Love', 'You', '!']
        # chars: [['I'], ['L', 'o', 'v', 'e'], ['Y', 'o', 'u'], ['!']]
        # label: [O, O, O, O]
        # Ids: 文档数 × 句子数 × [word_Ids, char_Ids, label_Ids, word_idx, d_idx]
        # word_Ids: word 在 alphabet 中的下标 [300, 20, 60, 180]
        # char_Ids: sentence_len × [char 在 alphabet 中的下标]
        # label_Ids: label 在 alphabet 中的下标
        # word_idx: word 在当前文档中的下标 [15, 16, 17, 18]
        # d_idx: 文档下标
        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []
        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        # token_num × token_embed
        # 跟 alphabet 对应
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        # 文档数 × 文档单词数 × max_memory_size
        # 根据数据集初始化后
        # word_mat[5][4] = [30, 40, ... , 0, 0, 0]
        # 表示第 6 篇文档中第 4 个单词在该文档从前往后出现的其他位置的位置下标
        # 文档从 0 开始计数，单词从 1 开始计数
        self.word_mat = []

    # 从命令行参数设置属性
    def read_config(self, args):
        self.train_dir = args.train_dir
        self.dev_dir = args.dev_dir
        self.test_dir = args.test_dir
        self.raw_dir = args.raw_dir
        self.model_dir = args.model_dir
        self.decode_dir = args.model_dir + "/results.txt"
        self.dset_dir = args.model_dir + "/data.dset"
        self.load_model_dir = args.model_dir + "/best_model.ckpt"

        self.save_model = str2bool(args.save_model)

        self.word_emb_dir = args.word_emb_dir
        self.norm_word_emb = str2bool(args.norm_word_emb)
        self.number_normalized = str2bool(args.number_normalized)

        self.status = args.status
        self.m1_iteration = int(args.m1_iteration)
        self.batch_size = int(args.batch_size)
        self.average_batch_loss = str2bool(args.ave_batch_loss)
        self.seed = int(args.seed)

        self.use_char = str2bool(args.use_char)
        self.char_emb_dim = int(args.char_emb_dim)
        self.char_feature_extractor = args.char_seq_feature
        self.char_hidden_dim = int(args.char_hidden_dim)
        self.word_emb_dim = int(args.word_emb_dim)
        self.wordRep_dropout = float(args.wordRep_dropout)

        self.bayesian_lstm_dropout = (float(args.bayesian_lstm_dropout), float(args.bayesian_lstm_dropout))
        self.model1_dropout = float(args.model1_dropout)
        self.hidden_dim = int(args.hidden_dim)
        self.model1_layer = int(args.model1_layer)
        self.bilstm = str2bool(args.bilstm)
        self.nsamples = int(args.nsample)
        self.threshold = float(args.threshold)

        self.max_read_memory = int(args.max_read_memory)

        self.clip_grad = float(args.clip_grad)
        self.l2 = float(args.l2)
        self.optimizer = args.optimizer
        self.lr = float(args.learning_rate)
        self.lr_decay = float(args.lr_decay)
        self.momentum = float(args.momentum)

    # 打印总览
    def show_data_summary(self):
        print("++"*50)
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % self.tagScheme)
        print("     MAX SENTENCE LENGTH: %s" % self.MAX_SENTENCE_LENGTH)
        print("     MAX   WORD   LENGTH: %s" % self.MAX_WORD_LENGTH)
        print("     Number   normalized: %s" % self.number_normalized)
        print("     Word  alphabet size: %s" % self.word_alphabet_size)
        print("     Char  alphabet size: %s" % self.char_alphabet_size)
        print("     Label alphabet size: %s" % self.label_alphabet_size)
        print("     Word embedding  dir: %s" % self.word_emb_dir)
        print("     Word embedding size: %s" % self.word_emb_dim)
        print("     Norm   word     emb: %s" % self.norm_word_emb)
        print("     Train  file directory: %s" % self.train_dir)
        print("     Dev    file directory: %s" % self.dev_dir)
        print("     Test   file directory: %s" % self.test_dir)
        print("     Raw    file directory: %s" % self.raw_dir)
        print("     Dset   file directory: %s" % self.dset_dir)
        print("     Model  file directory: %s" % self.model_dir)
        print("     Model loading directory: %s" % self.load_model_dir)
        print("     Decode file directory: %s" % self.decode_dir)
        print("     Train instance number: %s" % len(self.train_texts))
        print("     Dev   instance number: %s" % len(self.dev_texts))
        print("     Test  instance number: %s" % len(self.test_texts))
        print("     Raw   instance number: %s" % len(self.raw_texts))
        print("DATA SUMMARY END.")
        print("++"*50)
        sys.stdout.flush()

    # 初始化 Alphabet
    def build_alphabet(self, input_file):
        # 按行读取文件
        # in_lines：list of str
        in_lines = open(input_file, 'r').readlines()
        # 逐行处理
        for line in in_lines:
            # 字符数小于等于 2 的行是无效行，被跳过
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                # 老版本手动设置编码为 utf-8
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                    # 将 word 中的数字字符都换成 0
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.word_alphabet.add(word)
                self.label_alphabet.add(label)
                # 建立 char_alphabet
                for char in word:
                    self.char_alphabet.add(char)
        # 设置 alphabet_size
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

        # 判断标注策略
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():  # 字典迭代
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    # 固定 alphabet
    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    # 建立预训练 word_embedding
    # 提供了 embedding dir 就按文件内容建立，没有就不用建立
    # embedding dir 中文件内容是每行一个 embedding，一行由 token 和 token embedding 组成，空格分开
    # 没在文件中出现的 token，其 embedding 使用均匀分布生成
    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrain word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir,
                                                                                       self.char_alphabet,
                                                                                       self.char_emb_dim,
                                                                                       self.norm_char_emb)

    # 根据数据集，生成数据集相关信息
    def generate_instance(self, name):
        self.fix_alphabet()
        if len(self.word_mat) == 0:
            self.word_mat = []
            # 文档编号从 0 开始
            # 训练时，连续处理 3 个数据集，
            # 最终 doc_idx 是 3 个数据集总的文档数量
            # word_mat 也是总的结果
            self.doc_idx = 0
        if name == "train":
            self.train_texts, self.train_Ids, self.doc_idx, self.word_mat \
                = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet,
                                self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_mat,  self.doc_idx,
                                self.max_read_memory)
        elif name == "dev":
            self.dev_texts, self.dev_Ids,  self.doc_idx, self.word_mat \
                = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet,
                                self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_mat, self.doc_idx,
                                self.max_read_memory)
        elif name == "test":
            self.test_texts, self.test_Ids, self.doc_idx,  self.word_mat \
                = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet,
                                self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_mat,  self.doc_idx,
                                self.max_read_memory)
        elif name == "raw":
            self.raw_texts, self.raw_Ids, self.doc_idx, self.word_mat \
                = read_instance(self.raw_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet,
                                self.number_normalized, self.MAX_SENTENCE_LENGTH, self.word_mat,  self.doc_idx,
                                self.max_read_memory)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    # text: 文档数 × 句子数 × [words, features, chars, labels]
    # 转换成 [[words, features, chars, labels], ... , [words, features, chars, labels]]
    # 元素个数为数据集总句子数
    def convert_doc_to_sent(self, name):
        converted_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during converting, name should be within train/dev/test/raw !")
            exit(1)

        for doc in content_list:
            for sent in doc:
                converted_list.append(sent)

        if name == 'raw':
           self.raw_texts = converted_list
        elif name == 'test':
           self.test_texts = converted_list
        elif name == 'dev':
            self.dev_texts = converted_list
        elif name == 'train':
            self.train_texts = converted_list

    # predict_results: [[一个句子的预测结果]]，元素个数为总句子数
    # 生成预测结果文件
    def write_decoded_results(self, predict_results, name):
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        # 运行这个之前，运行了 convert_doc_to_sent
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        # 生成预测结果文件
        # 一行为 word + ' ' + label
        # 每句话空行隔开
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                # content_list[idx] is a list with [words, features, chars, labels]
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, self.decode_dir))

    # 生成预测结果文件，在 word 和 pre_label 中添加了 gold_label
    # 这个方法本项目中并未用到
    def write_decoded_results_with_golds(self, predict_results, gold_results, name):
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                fout.write(
                    content_list[idx][0][idy] + " " + gold_results[idx][idy] + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    # 文件反序列化成 Data 对象
    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    # Data 对象序列化成文件
    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)  # 序列化对象，2 是控制使用的协议
        f.close()


# 配置文件转字典
def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":  # 注释行
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated."%(pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(str):
    if type(str) is bool:
        return str
    if str == "True" or str == "true" or str == "TRUE":
        return True
    else:
        return False
