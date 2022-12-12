import numpy as np
from queue import Queue
import torch

if __name__ == '__main__':
    a = np.array([[1, 2, 3], [1, 2, 3]])
    b = np.array([[1], [2]])
    c = np.append(a, b, axis=1)
    print(c)
    for i in np.vsplit(c, 2):
        print(i.flatten())

    history = []
    l = 2
    history.append(a)
    if len(history) > l:
        history = history[1:l+1]
    print(history)
    history.append(b)
    if len(history) > l:
        history = history[1:l+1]
    print(history)
    history.append(c)
    if len(history) > l:
        history = history[1:l+1]
    print(history)
    history = []
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    history.append(a)
    history.append(b)
    print(history)
    print(np.array(history))
    history = []
    history = torch.FloatTensor(np.array(history)).unsqueeze(0)
    print(history)
