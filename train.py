import logging
import torch
import os

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from WebVision.dataloader import WebVisionImageDataset



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
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")

    # Init CNN
    if args.model == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes=args.classes)

    net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    net.train()

    # Init Dataset
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    wv_dataset = WebVisionImageDataset(args, transform=train_transform)
    trainloader = DataLoader(wv_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    val_dataset = WebVisionImageDataset(args, train=False, val=True, test=False, transform=test_transform)
    validateloader = DataLoader(val_dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):
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
            # if step % 1000 == 999:  # print every 2000 mini-batches
            if True:  # Debug
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 10))
                running_loss = 0.0
                # Acc on Val
                net.eval()
                validation(net, validateloader, device, step)
                net.train()

    if args.save:
        checkpoint = "%s_%d.pt" % (args.model, args.epochs)
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


if __name__ == '__main__':
    print("Hello")