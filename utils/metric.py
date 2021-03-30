from __future__ import print_function


# 输入：总句子数 × [O O O O]
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    # 句子数 × 命名实体列表
    golden_full = []
    predict_full = []
    right_full = []

    right_tag = 0
    all_tag = 0

    for idx in range(0, sent_num):  # 按句子遍历
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1  # 正确标签数
        all_tag += len(golden_list)
        # 获得结果中的命名实体
        if label_type == "BMES":
            # 获得形如 ['[0,1]PER', '[2,2]LOC', '[4,6]LOC', '[8,8]LOC'] 这样的返回
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        # 正确命名实体 -- 取交集
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    # 命名实体数
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    # 精确率计算
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num+0.0)/predict_num
    # 召回率计算
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    # F1 值计算
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    # 按 token 来看的准确率
    accuracy = (right_tag+0.0)/all_tag
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


# 输入：[O O O O]
# 输出：命名实体列表，
# 格式如 -- ['[0,1]PER', '[2,2]LOC', '[4,6]LOC', '[8,8]LOC', '[14,15]MISC', '[23,23]LOC']
def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):  # 按句子单词数遍历
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            # 将旧字符串替换成新字符串，替换次数为 1
            # 即删除开头的 'B-'
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            # 'PER[0,1]' 转换成 '[0,1]PER'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


# 同上，只是策略不同
def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


# 从 input_file 获得 sentences 和 labels
# sentences -- [sentence]
# sentence -- [token]
# labels -- [label]
# label -- [token_label]
def readSentence(input_file):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences, labels


# 从文件中读标签
# 文件内容为一行代表一个单词，第一列是 token，第二列是 gold_label，最后一列是 pred_label
# 返回 sentences, golden_labels, predict_labels
def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file, 'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:  # 一句话结束
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences, golden_labels, predict_labels


# golden_label 和 predict_label 在不同文件
def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file:", golden_file, predict_file)
    print("Label format:", label_type)
    golden_sent, golden_labels = readSentence(golden_file)
    predict_sent, predict_labels = readSentence(predict_file)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print("P:%sm R:%s, F:%s" % (P, R, F))


# golden_label 和 predict_label 在同一文件
def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent, golden_labels, predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print("P:%s, R:%s, F:%s" % (P, R, F))


if __name__ == '__main__':
    print(
        get_ner_BIO(
            'B-PER I-PER B-LOC O B-LOC I-LOC I-LOC O B-LOC O O O O O '
            'B-MISC I-MISC O O O O O O O B-LOC O O O O O O O O O'.split(' ')))
    # print "sys:",len(sys.argv)
    # if len(sys.argv) == 3:
    #     fmeasure_from_singlefile(sys.argv[1], "BMES", int(sys.argv[2]))
    # else:
    #     fmeasure_from_singlefile(sys.argv[1], "BMES")

