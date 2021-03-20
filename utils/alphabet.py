import sys


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        # 两个数据结构，一个 list 存储单词，一个 dict，存储单词在 list 中的下标
        self.instance2index = {}  # index 是 instance 在 instance 中的下标
        self.instances = []
        self.keep_growing = keep_growing

        # 位置 0 默认都是被占用的，因为相关 tensor 中 0 值认为是无效位
        self.default_index = 0
        self.next_index = 1  # 标识下一个空位

        # label 的字母表中不用添加 UNKNOWN 单词，因为标签 O 等价于 UNKNOWN
        if not self.label:
            # 除了 label 字母表，其他字母表的第一个单词都是 UNKNOWN
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        self.default_index = 0
        self.next_index = 1

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            # 在还在增长的时候，遇到 OOV，要添加
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            if self.label:
                return self.instances[0]
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING: Alphabet get_instance, unknown instance, return the first label.')
            return self.instances[0]

    def size(self):
        # +1 是因为下标从 1 开始
        return len(self.instances) + 1

    # 字典迭代
    def iteritems(self):
        if sys.version_info[0] < 3:  # If using python3, dict item access uses different syntax
            return self.instance2index.iteritems()
        else:
            return self.instance2index.items()

    # [index:instance]
    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}
