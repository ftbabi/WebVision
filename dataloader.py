import yaml
import json
import os
import logging
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchvision
from skimage import io
from PIL import Image
import numpy as np
from devkit.webvision.config import LoadInfo, LoadTrain, LoadVal, LoadTest, LoadQuerySynsetMap

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class WebVisionImageDataset(Dataset):
    """
    WebVision Images
    """
    def __init__(self, args, train=True, val=False, test=False, transform=None, use_class_id=False):
        """

        :param dataframe: If google, only google; If all, concat
        :param transform:
        """
        self.df = self.__load_dataset(args, train, val, test)
        self.transform = transform
        self.use_class_id = use_class_id
        self.train = train
        self.classes = args.classes

    def __load_dataset(self, args, train, val, test):
        # info = LoadInfo()
        if train:
            # info = LoadTrain(args.filter)
            # train_content = info.loc[info['type'] == 'train']
            # train_content = info
            ans = LoadTrain(args.filter, args.traindata, args.subset, args.random)
            # if args.traindata == 'google':
            #     ans = train_content.loc[train_content['source'] == 'google', :]
            # elif args.traindata == 'flickr':
            #     ans = train_content.loc[train_content['source'] == 'flickr', :]
            # elif args.traindata == 'all':
            #     ans = train_content
            # else:
            #     print("No train data set specified")
            #     exit(-1)

            if args.filter:
                ans = ans.loc[ans['selected'] == '1', :]
                if len(ans) <= 1:
                    print("Filter key invalid, not \'true\'")
                    exit(-1)
        elif val:
            info = LoadVal()
            # ans = info.loc[info['type']=='val']
            ans = info
        elif test:
            info = LoadTest()
            # ans = info.loc[info['type']=='test']
            ans = info

        if args.checkimg:
            ans = self.__checkvalid(ans)
        print("%d images are used for training" % len(ans))

        return ans

    def __checkvalid(self, info):
        invalid_files_idx = []
        invalid_files_path = []
        for index, row in info.iterrows():
            if not os.path.exists(row['image_path']):
                invalid_files_idx.append(index)
                invalid_files_path.append(row['image_path'])

        if invalid_files_idx:
            print("Warning: %d files don't exist" % len(invalid_files_idx))
            logging.warning("Warning: %d files don't exist" % len(invalid_files_idx))
            for file in invalid_files_path:
                logging.warning("==> %s" % file)
            info.drop(invalid_files_idx, inplace=True)

        return info



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_line = self.df.iloc[idx, :]
        img_path = img_line.loc['image_path']
        if self.train and self.use_class_id:
            img_label_idx = int(img_line.loc['class_id']) - 1
        else:
            img_label_idx = int(img_line.loc['label'])
        # img_label = np.eye(self.classes)[img_label_idx]
        # img_label[img_label_idx] = 1
        img_label = img_label_idx
        # try:
        #     # image = io.imread(img_path)
        #     image = Image.open(img_path).convert('RGB')  # PIL.Image.Image对象
        #     # img_pil_1 = np.array(img_pil)
        #     # print(type(image))
        #
        #     if self.transform:
        #         image = self.transform(image)
        # except:
        #     print(img_path)
        #     exit(-1)

        # image = io.imread(img_path)
        image = Image.open(img_path).convert('RGB')  # PIL.Image.Image对象
        # img_pil_1 = np.array(img_pil)
        # print(type(image))

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'label': img_label}
        return image, img_label

class MetaLoader:
    def __init__(self, metapath, config_path='./config.yaml'):
        # Load configuration
        # f = open(config_path)
        # content = yaml.load(f)
        # f.close()

        # Load meta
        prefix, type =  os.path.splitext(metapath)
        if type == '.json':
            with open(metapath, 'r') as f:
                self.meta = json.load(f)
        elif type == '.txt':
            with open(metapath, 'r') as f:
                self.meta = f.readlines()

        # Version
        self.version = 'google'
        if 'flickr' in metapath:
            self.version = 'flickr'

    def getData(self):
        return self.meta

class QuerySynsetMapLoader:
    def __init__(self, map_path):
        # Load map
        with open(map_path, 'r') as f:
            lines = f.readlines()
        # Convert to dict
        self.query2synset = {}
        self.synset2query = {}
        for line in lines:
            line = line.split(' ')
            self.query2synset[int(line[0])] = int(line[1])
            if int(line[1]) in self.synset2query.keys():
                self.synset2query[int(line[1])].append(int(line[0]))
            else:
                self.synset2query[int(line[1])] = [int(line[0])]

    def getSynsetQueryMap(self):
        return self.synset2query

    def getQuerySynsetMap(self):
        return self.query2synset

if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
