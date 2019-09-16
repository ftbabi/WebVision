import logging
import torch
import os
import tqdm

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn


from torch.utils.data import DataLoader
from WebVision.devkit.webvision.config import LoadVal, VAL_DIVIDE_FOLDER
from WebVision.dataloader import WebVisionImageDataset
from WebVision.train import validation



def evaluate(args):
    # select optimizer
    if not args.load:
        print("Warning: No model checkpoint selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpath = args.load
    print("Load model: ", checkpath)
    logging.info("Load model: %s" % checkpath)
    modelfile = torch.load(checkpath, map_location=device)
    modelname = os.path.basename(checkpath)
    modelname = modelname[:modelname.rfind('_')]
    if modelname == 'resnext50_32x4d':
        print("Model: %s" % modelname)
        logging.info("Model: %s" % modelname)
        net = models.resnext50_32x4d(num_classes=args.classes)
        # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

        net.module.load_state_dict(modelfile['model_state_dict'])
        # optimizer.load_state_dict(modelfile['optimizer_state_dict'])

        net.to(device)
    else:
        print("UNK: %s" % modelname)
        logging.info("UNK: %s" % modelname)
        exit(-1)

    print("Test begin...")
    logging.info("Test begin...")
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    testdataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    testloader = DataLoader(testdataset, batch_size=args.batch_size)

    validation(net, testloader, device, 0)

def divide_val_img():
    val_data = LoadVal()
    prev_class = -1
    class_dir = ''
    for index, row in val_data.iterrows():
        cur_class = row['label']
        if cur_class != prev_class:
            prev_class = cur_class
            class_dir = os.path.join(VAL_DIVIDE_FOLDER, str(cur_class))
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
        src_path = row['image_path']
        src_file = os.path.basename(src_path)
        dst_path = class_dir
        dst_file = os.path.join(dst_path, src_file)
        if not os.path.exists(dst_file):
            cmd = 'cp %s %s' % (src_path, dst_path)
            ret = os.popen(cmd)
            ret.read()
            print(cmd)




if __name__ == '__main__':
    divide_val_img()