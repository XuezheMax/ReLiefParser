__author__ = 'max'

import numpy as np

NEGTIVE_REWARD = -1


class Environment(object):
    def __init__(self, heads, marks):
        assert heads.ndim == 2
        assert marks.ndim == 2
        self.__batch_size = heads.shape[0]
        self.__length = heads.shape[1]
        self.__sizes = marks.sum(axis=1)
        self.__marks = np.copy(marks)
        self.__rewards = np.zeros((self.__batch_size, self.__length, self.__length), dtype=np.float32)
        for b in range(self.__batch_size):
            self.__rewards[b, 0] = NEGTIVE_REWARD
            for i in range(1, self.__length):
                if self.__marks[b, i]:
                    self.__rewards[b, i] = (1 - self.__marks[b]) * NEGTIVE_REWARD
                    self.__rewards[b, i, heads[b, i]] = 1.0
                else:
                    self.__rewards[b, i] = NEGTIVE_REWARD

    def take_action(self, acts):
        # each act is between [0, 2n-1]
        assert acts.shape == (self.__batch_size,)
        assert (acts >= 0).all()
        assert (acts < 2 * self.__sizes).all()

        # [0, n-1] as left acts, [n, 2n-1] as right acts
        is_lefts = acts < self.__sizes
        heads = acts % self.__sizes
        children = np.zeros_like(heads)
        rewards = np.zeros_like(heads, dtype=np.float32)

        for b in range(self.__batch_size):
            head = heads[b]
            child = self.__find_left(b, head) if is_lefts[b] else self.__find_right(b, head)
            children[b] = child
            if self.__marks[b, head]:
                if child >= 0:
                    rewards[b] = self.__rewards[b, child, head]
                    self.__marks[b, child] = 0
                else:
                    rewards[b] = NEGTIVE_REWARD
            else:
                rewards[b] = NEGTIVE_REWARD
                if child >= 0:
                    self.__marks[b, child] = 0

        return (rewards, heads, children)

    def __find_left(self, b, pos):
        left = -1
        for i in range(pos - 1, -1, -1):
            if self.__marks[b, i]:
                left = i
                break
        return left

    def __find_right(self, b, pos):
        right = -1
        for i in range(pos + 1, self.__sizes[b]):
            if self.__marks[b, i]:
                right = i
                break
        return right

    def display(self):
        print self.__length
        print self.__sizes
        print self.__marks
        print self.__rewards
