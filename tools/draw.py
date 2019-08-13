import matplotlib.pyplot as plt
import numpy as np
import json


def draw_orig(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    keys = []
    axis = []
    for order, line in enumerate(lines):
        axis.append(order)
        keys.append(float(line))

    plt.scatter(axis, keys)
    plt.show()

def draw_count(path, acc=1):
    with open(path, 'r') as f:
        lines = f.readlines()
    dt = {}
    for order, line in enumerate(lines):
        num = float(line.split('\n')[0])*acc
        if int(num) in dt.keys():
            dt[int(num)] += 1
        else:
            dt[int(num)] = 1

    plt.scatter(dt.keys(), dt.values())
    plt.show()

def testjson():
    a = np.array([1,2,3])
    b = np.array([3,4,5])
    data = {0:[a], 1:[a,b]}
    filename = "jsontest.json"
    with open(filename, 'w') as f:
        json.dump(data, f)

    with open(filename, 'r') as f:
        reload = json.load(f)
        print(reload)

if __name__ == '__main__':
    path = '../NLP/draw.txt'
    # draw_orig(path)
    # draw_count(path, 1)
    testjson()