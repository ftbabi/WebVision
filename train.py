import logging
import torch
import os
import time
import numpy as np

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
from dataloader import WebVisionImageDataset, LoadQuerySynsetMap
from extract_features import FeatureExtractor
from tqdm import tqdm

def mean_std():
    # Init Dataset
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    mean = [0.502, 0.476, 0.428]
    std = [0.299, 0.296, 0.309]

    return mean, std


def train(args):
    """
    TODO:
        Additional choices: criterion, optimizer

    :param args:
    :return:
    """
    logging.info("Begin training")
    print("Begin training")

    # Init GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init CNN
    cur_epoch = 0
    if args.load:
        net, _, cur_epoch = loadmodel(args, device)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'resnext50_32x4d':
        if args.pretrained:
            net = models.resnext50_32x4d(pretrained=args.pretrained)
            if args.classes != 1000:
                net.fc = nn.Linear(2048, args.classes)
        else:
            net = models.resnext50_32x4d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'resnext101_32x8d':
        if args.pretrained:
            net = models.resnext101_32x8d(pretrained=args.pretrained)
            if args.classes != 1000:
                net.fc = nn.Linear(2048, args.classes)
        else:
            net = models.resnext101_32x8d(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'resnet50':
        if args.pretrained:
            net = models.resnet50(pretrained=args.pretrained)
            if args.classes != 1000:
                net.fc = nn.Linear(2048, args.classes)
        else:
            net = models.resnet50(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.model == 'alexnet':
        net = models.alexnet(pretrained=args.pretrained, num_classes=args.classes)
        if args.pretrained and args.classes != 1000:
            net.fc = nn.Linear(4096, args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        print("UNK model %s" % (args.model))
        exit(-1)

    print("Using model %s" % (args.model))

    # Init optim
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.weightschedule, gamma=0.1)

    # Allocate CNN on GPUs
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()

    mean, std = mean_std()

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.model == 'alexnet':
        train_transform = transforms.Compose(
            [transforms.Resize(256), transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(227), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(227), transforms.ToTensor(),
             transforms.Normalize(mean, std)])


    print("Loading training data")
    wv_dataset = WebVisionImageDataset(args, train=True, transform=train_transform, use_class_id=args.classid_as_label)
    trainloader = DataLoader(wv_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.ngpu*args.thread)

    print("Loading validation data")
    val_dataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    validateloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.ngpu*args.thread)

    query_synset_map = {}
    if args.classid_as_label:
        print("Loading query synset map")
        query_synset_map = LoadQuerySynsetMap()


    globalstep = 0
    globalepoch = 0
    running_loss = 0.0

    # print("Before training...")
    # net.eval()
    # validation(net, validateloader, device, 0, query_synset_map)
    # net.train()

    print("Current epoch is %d, lr: %f" % (cur_epoch+1, optimizer.param_groups[0]['lr']))
    # scheduler.step(61)

    for epoch in range(cur_epoch, args.epochs):
        print("Epoch %d begin" % (epoch + 1))
        globalepoch = epoch
        running_loss = 0.0
        start = time.time()
        iterstamp = start
        num_samples, num_correct = 0, 0
        for step, data in enumerate(trainloader, 0):
            globalstep = step
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # labels = np.eye(args.classes)[labels]
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # debug_flag = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Training acc
            _, preds = torch.max(outputs, 1)
            num_samples += labels.size(0)
            num_correct += (preds == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if (step % args.printfreq) == (args.printfreq - 1):  # print every args.printfreq mini-batches
                iter_curtime = time.time()
                print('[%3d, %5d] Time cost: %d; Loss: %.3f; Acc: (%d/%d) %.3f' %
                      (epoch + 1, step + 1, iter_curtime - iterstamp, running_loss / args.printfreq,
                       num_correct, num_samples, num_correct/num_samples))
                logging.info('[%3d, %5d] Time cost: %d; Loss: %.3f; Acc: (%d/%d) %.3f' %
                      (epoch + 1, step + 1, iter_curtime - iterstamp, running_loss / args.printfreq,
                       num_correct, num_samples, num_correct/num_samples))
                running_loss = 0.0
                iterstamp = iter_curtime
                # Acc on Val
            if (step % args.valfreq) == (args.valfreq - 1):
                # eval
                net.eval()
                validation(net, validateloader, device, step, query_synset_map)
                net.train()
        # Epoch val
        net.eval()
        validation(net, validateloader, device, step, query_synset_map)
        net.train()

        # For the next epoch
        # Therefore, when resume, starts from 30, the next 30 iters doesn't update lr; Should be initial.
        # E.g., load model 60. initial with 0.001(cause 0-29 0.1, 30-59 0.01, 60- 0.001)
        scheduler.step(epoch+1-cur_epoch)
        during = time.time() - start
        print("Epoch %d, Cost: %d, Lr: %f" % (epoch+1, during, optimizer.param_groups[0]['lr']))
        logging.info("Epoch %d, Cost: %d, Lr: %f" % (epoch+1, during, optimizer.param_groups[0]['lr']))
        if epoch % 10 == 9 and args.save:
            print("Epoch %d save model..." % (epoch+1))
            savemodel(args, net, optimizer, epoch+1)

    print('[%d, %5d] loss: %.3f' %
          (globalepoch + 1, globalstep + 1, running_loss / 10))

    # Acc on Val
    net.eval()
    validation(net, validateloader, device, globalstep)
    net.train()

    if args.save:
        savemodel(args, net, optimizer, args.epochs)

    return net

def savemodel(args, net, optimizer, epoch):
    checkpoint = "%s-%s-%d-%d.pt" % (args.model, args.traindata, epoch, args.subset)
    checkpath = os.path.join(args.save, checkpoint)
    print("Save model: %s" % checkpath)
    logging.info("Save model: %s" % checkpath)
    torch.save({'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }
               , checkpath)

def querySynsetMapFunc(data, query_synset_map):
    if not query_synset_map:
        return data

    ans = []
    for i in data:
        # print(i)
        key = i.item()
        if str(key+1) not in query_synset_map.keys():
            print("Error: %d not exsit " % (key+1))
        label = query_synset_map[str(key+1)]
        ans.append(int(label))

    ans = torch.Tensor(ans)
    return ans


def validation(net, validateloader, device, step, query_synset_map={}):
    start_time = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validateloader:
            inputs, labels = data
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = querySynsetMapFunc(predicted, query_synset_map)
            predicted = predicted.long()
            predicted = predicted.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    during = time.time() - start_time
    print('Iter: %d, Accuracy of the network on the validation set: %f %% %d/%d, Val time cost: %d' % (
                step, 100 * correct / total, correct, total, during))
    logging.info('Iter: %d, Accuracy of the network on the validation set: %f %% %d/%d, Val time cost: %d' % (
                step, 100 * correct / total, correct, total, during))



def loadmodel(args, device):
    checkpath = args.load
    print("Load model: ", checkpath)
    logging.info("Load model: %s" % checkpath)
    modelfile = torch.load(checkpath, map_location=device)
    filepath = os.path.basename(checkpath)
    filename, type = os.path.splitext(filepath)
    # modelname = modelname[:modelname.find('-')]
    split_filename = filename.split('-')
    modelname = split_filename[0]
    epoch = split_filename[2]
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
    elif modelname == 'resnet50':
        print("Model: %s" % modelname)
        logging.info("Model: %s" % modelname)
        net = models.resnet50(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

        net.load_state_dict(modelfile['model_state_dict'])
        optimizer.load_state_dict(modelfile['optimizer_state_dict'])
    elif modelname == 'alexnet':
        print("Model: %s" % modelname)
        logging.info("Model: %s" % modelname)
        net = models.alexnet(num_classes=args.classes)
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

        net.load_state_dict(modelfile['model_state_dict'])
        optimizer.load_state_dict(modelfile['optimizer_state_dict'])
    else:
        print("UNK: %s" % modelname)
        logging.info("UNK: %s" % modelname)
        exit(-1)

    # filename =

    return net, optimizer, int(epoch)

def evaluate(args):
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

    net, _, _ = loadmodel(args, device)
    # Allocate CNN on GPUs
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    print("Test begin...")
    logging.info("Test begin...")
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    testdataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, num_workers=args.ngpu*args.thread)

    validation(net, testloader, device, 0)

def extract_features(args):
    """
    Only for training data
    :param args:
    :return:
    """

    # Init GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.load:
        print("No model provided")
        logging.info("No model provided")
        exit(-1)
    if not args.feature_dir:
        print("No feature output dir")
        logging.info("No feature output dir")
        exit(-1)

    modelname = os.path.basename(args.load)
    filename, type = os.path.splitext(modelname)
    filename = filename.split('-')
    if len(filename) < 4:
        print("Error in splitting filename: %s" % (modelname))
        exit(-1)
    feature_name = "%s-%s-%s-%d.txt" % (filename[0], args.traindata, filename[2], args.subset)
    feature_path = os.path.join(args.feature_dir, feature_name)

    mean, std = mean_std()

    net, _, _ = loadmodel(args, device)
    # Allocate CNN on GPUs
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    print("Extract features begin...")
    logging.info("Extract features begin...")
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset = WebVisionImageDataset(args, train=True, val=False, test=False, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.ngpu*args.thread, shuffle=False)

    start_time = time.time()
    features = []
    # net.eval()
    extract_result = FeatureExtractor(net)
    extract_result.eval()

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = extract_result(inputs)
            x = outputs[-2]
            x = x.view(x.size(0), -1)
            features.append(x)
            # np.savetxt(feature_path, features, fmt='%f', delimiter=',')

    during = time.time() - start_time
    print("Extraction time cost %d" % (during))
    logging.info("Extraction time cost %d" % (during))
    features = np.array(features)
    np.savetxt(feature_path, features, fmt='%f', delimiter=',')
    print("Save to %s" % (feature_path))
    logging.info("Save to %s" % (feature_path))


if __name__ == '__main__':
    print("Hello")
    a = 'ab-cd'
    b = a
    c = a.split('-')

    print(a)
    print(b)
    print(c)
    print(a[:a.find('-')])