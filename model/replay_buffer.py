class ReplayBuffer(object):
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

