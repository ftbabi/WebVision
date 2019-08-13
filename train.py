import logging
import torch
import os
import time

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from WebVision.dataloader import WebVisionImageDataset


def mean_std():
    # Init Dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return mean, std


def train(args):
    """
    TODO:
        Additional choices: criterion, optimizer

    :param args:
    :return:
    """
    logging.info("Begin training")

    # Init GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init CNN
    if args.load:
        net, _ = loadmodel(args, device)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'resnext101_32x8d':
        net = models.resnext101_32x8d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

    # Allocate CNN on GPUs
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()

    # Init Dataset
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    mean, std = mean_std()

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

    print("Loading training data")
    wv_dataset = WebVisionImageDataset(args, transform=train_transform)
    trainloader = DataLoader(wv_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # wv_dataset = 1
    # trainloader = 1

    print("Loading validation data")
    val_dataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    validateloader = DataLoader(val_dataset, batch_size=args.batch_size)

    globalstep = 0
    globalepoch = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        print("Epoch %d begin" % epoch)
        globalepoch = epoch
        running_loss = 0.0
        start = time.time()
        for step, data in enumerate(trainloader, 0):
            globalstep = step
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 1000 == 999:  # print every 2000 mini-batches
            # if True:  # Debug
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 10))
                running_loss = 0.0
                # Acc on Val
                net.eval()
                validation(net, validateloader, device, step)
                net.train()
        during = time.time() - start
        print("Epoch %d cost: %d" % (epoch, during))
        logging.info("Epoch %d cost: %d" % (epoch, during))

    print('[%d, %5d] loss: %.3f' %
          (globalepoch + 1, globalstep + 1, running_loss / 10))

    # Acc on Val
    net.eval()
    validation(net, validateloader, device, globalstep)
    net.train()

    if args.save:
        checkpoint = "%s-%s-%d.pt" % (args.model, args.traindata, args.epochs)
        checkpath = os.path.join(args.save, checkpoint)
        print("Save model: %s" % checkpath)
        logging.info("Save model: %s" % checkpath)
        torch.save({'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }
                   , checkpath)

    return net

def validation(net, validateloader, device, step):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validateloader:
            inputs, labels = data
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Iter: %d, Accuracy of the network on the validation set: %f %%' % (
                step, 100 * correct / total))
    logging.info('Iter: %d, Accuracy of the network on the validation set: %f %%' % (
                step, 100 * correct / total))

def loadmodel(args, device):
    checkpath = args.load
    print("Load model: ", checkpath)
    logging.info("Load model: %s" % checkpath)
    modelfile = torch.load(checkpath, map_location=device)
    modelname = os.path.basename(checkpath)
    modelname = modelname[:modelname.find('-')]
    if modelname == 'resnext50_32x4d':
        print("Model: %s" % modelname)
        logging.info("Model: %s" % modelname)
        net = models.resnext50_32x4d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

        net.load_state_dict(modelfile['model_state_dict'])
        optimizer.load_state_dict(modelfile['optimizer_state_dict'])

        # net.to(device)
    elif modelname == 'resnext101_32x8d':
        print("Model: %s" % modelname)
        logging.info("Model: %s" % modelname)
        net = models.resnext101_32x8d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

        net.load_state_dict(modelfile['model_state_dict'])
        optimizer.load_state_dict(modelfile['optimizer_state_dict'])
    else:
        print("UNK: %s" % modelname)
        logging.info("UNK: %s" % modelname)
        exit(-1)

    return net, optimizer

def evaluate(args):
    # TODO:
    # select optimizer
    if not args.load:
        print("Warning: No model checkpoint selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpath = args.load
    # print("Load model: ", checkpath)
    # logging.info("Load model: %s" % checkpath)
    # modelfile = torch.load(checkpath, map_location=device)
    # modelname = os.path.basename(checkpath)
    # modelname = modelname[:modelname.rfind('_')]
    # if modelname == 'resnext50_32x4d':
    #     print("Model: %s" % modelname)
    #     logging.info("Model: %s" % modelname)
    #     net = models.resnext50_32x4d(num_classes=args.classes)
    #     # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    #
    #     net.module.load_state_dict(modelfile['model_state_dict'])
    #     # optimizer.load_state_dict(modelfile['optimizer_state_dict'])
    #
    #     net.to(device)
    # else:
    #     print("UNK: %s" % modelname)
    #     logging.info("UNK: %s" % modelname)
    #     exit(-1)

    mean, std = mean_std()

    net, _ = loadmodel(args, device)
    net.to(device)

    print("Test begin...")
    logging.info("Test begin...")
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    testdataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    testloader = DataLoader(testdataset, batch_size=args.batch_size)

    validation(net, testloader, device, 0)


if __name__ == '__main__':
    print("Hello")
    a = 'ab-cd'
    print(a[:a.find('-')])