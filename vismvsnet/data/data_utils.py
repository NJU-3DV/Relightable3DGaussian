import numpy as np


def cycle(iterable, num_cycle):
    if num_cycle != 0:
        for i in range(num_cycle):
            for it in iterable:
                yield it
    else:
        while True:
            for it in iterable:
                yield it


class Cycle:

    def __init__(self, iterable, num_cycle):
        self.iterable = iterable
        self.num_cycle = num_cycle
        self.iterator = cycle(iterable, num_cycle)

    def __iter__(self):
        return self

    def __next__(self):
        return self.iterator.__next__()

    def __len__(self):
        if self.num_cycle == 0:
            raise ValueError('Cannot get length of infinite iterator.')
        else:
            return len(self.iterable) * self.num_cycle


class Until:

    def __init__(self, iterable, total_steps):
        self.iterable = iterable
        self.total_steps = total_steps
        self.count = 0
        self.iterator = cycle(iterable, 0)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.total_steps:
            raise StopIteration
        self.count += 1
        return self.iterator.__next__()

    def __len__(self):
        return self.total_steps


def numpy_collate(data_):
    transpose = zip(*data_)
    # for d in data_:
    #     for arr in d:
    #         if type(arr) == list:
    #             print([a.shape for a in arr])
    #         else:
    #             print(arr.shape)
    return [np.stack(batch, axis=0) for batch in transpose]


def dict_collate(data_):
    return {k: np.stack([d[k] for d in data_], axis=0) for k in data_[0]}
