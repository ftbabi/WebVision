import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import logging
from WebVision.train import train, evaluate

def parse():
    parser = argparse.ArgumentParser(description='Training on WebVision',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    # parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('--traindata', type=str, choices=['google', 'flickr', 'all'], help='Choose among google/flickr/all.', default='all')

    # Data agumentation
    parser.add_argument('--transform', type=str, choices=['crop'],
                        help='Choose among crop.', default='crop')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data', default=False)

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    # parser.add_argument('--test_bs', type=int, default=10)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
    #                     help='Decrease learning rate at these epochs.')
    # parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
    # parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

    # Architecture
    parser.add_argument('--classes', type=int, default=5000, help='Number of classes.')
    parser.add_argument('--model', type=str, choices=['resnext50_32x4d', 'resnext101_32x8d', 'densenet121'],
                        help='Choose among resnext50_32x4d/densenet121.', default='resnext101_32x8d')

    # parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    # parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    # parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    # parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    # parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')

    # Choices
    parser.add_argument('--train', action='store_true', help='train the model', default=False)
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set', default=False)
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model', default=False)

    # Train config


    # Parse
    args = parser.parse_args()

    return args

# test function (forward only)
def test():
        # net.eval()
        # loss_avg = 0.0
        # correct = 0
        # for batch_idx, (data, target) in enumerate(test_loader):
        #     data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        #
        #     # forward
        #     output = net(data)
        #     loss = F.cross_entropy(output, target)
        #
        #     # accuracy
        #     pred = output.data.max(1)[1]
        #     correct += float(pred.eq(target.data).sum())
        #
        #     # test loss average
        #     loss_avg += float(loss)
        #
        # state['test_loss'] = loss_avg / len(test_loader)
        # state['test_accuracy'] = correct / len(test_loader.dataset)
    pass

def init(args):
    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    logfile = os.path.join(args.log, 'webvision_cli.log')
    if not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.write('')

    logging.basicConfig(level=logging.INFO,  # 定义输出到文件的log级别，
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                        filename=logfile,  # log文件名
                        filemode='a')
    state = {k: v for k, v in args._get_kwargs()}
    logging.info(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    # args.epochs = args.epochs * 128 // args.batch_size
    # args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    # if not os.path.isdir(args.data_path):
    #     os.makedirs(args.data_path)

    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #
    # train_transform = transforms.Compose(
    #     [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
    #      transforms.Normalize(mean, std)])
    # test_transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Init checkpoints
    # if not os.path.isdir(args.save):
    #     os.makedirs(args.save)

    # Init model, criterion, and optimizer
    # net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    # print(net)
    # if args.ngpu > 1:
    #     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    #
    # if args.ngpu > 0:
    #     net.cuda()
    #
    # optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
    #                                 weight_decay=state['decay'], nesterov=True)
    # Main loop
    # best_accuracy = 0.0
    # for epoch in range(args.epochs):
    #     if epoch in args.schedule:
    #         state['learning_rate'] *= args.gamma
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = state['learning_rate']
    #
    #     state['epoch'] = epoch
    #     train()
    #     test()
    #     if state['test_accuracy'] > best_accuracy:
    #         best_accuracy = state['test_accuracy']
    #         torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
    #     log.write('%s\n' % json.dumps(state))
    #     log.flush()
    #     print(state)
    #     print("Best accuracy: %f" % best_accuracy)

if __name__ == '__main__':
    args = parse()

    # args.traindata = 'google'
    # args.batch_size = 4
    args.train = True
    args.save = './checkpoints'
    # args.load = './checkpoints/resnext50_32x4d-1.pt'

    init(args)


    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.prefetch, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
    #                                           num_workers=args.prefetch, pin_memory=True)

    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)








