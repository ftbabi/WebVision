import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    path = '../NLP/draw.txt'
    # draw_orig(path)
    draw_count(path, 1)