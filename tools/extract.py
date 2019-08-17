import os
import json
import re
import sys
import subprocess
from tqdm import tqdm
from threading import Thread
import time
import matplotlib.pyplot as plt

def test():
    # srcdir = '/home/shaoyidi/Downloads'
    # tardir = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/data/2018/train_images_256'
    srcdir = '/home/ydshao/VirtualProjects/WebVision/data/2018/downloads'
    tardir = '/home/ydshao/VirtualProjects/WebVision/data/2018/train_images_256'
    if not os.path.exists(tardir):
        print("%s not exist" % tardir)
        exit(-1)
    filelist = []
    cmdlist = []
    for i in range(8, 33):
        filename = 'webvision_train_%02d.tar' % i
        filepath = os.path.join(srcdir, filename)
        filelist.append(filepath)
        # if not os.path.exists(filepath):
        #     print("Warning")
        cmd = 'tar xvf %s -C %s' % (filepath, tardir)
        cmdlist.append(cmd)


    for cmd in tqdm(cmdlist):
        print(cmd)
        ret = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # ret.wait()
        info = ret.stdout.readlines()
        # print(info)

    print("finish")


if __name__ == '__main__':
    test()