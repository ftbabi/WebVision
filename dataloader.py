import yaml
import json
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchvision
from skimage import io
from PIL import Image
import numpy as np
from WebVision.devkit.webvision.config import LoadInfo, LoadTrain, LoadVal, LoadTest


class WebVisionImageDataset(Dataset):
    """
    WebVision Images
    """
    def __init__(self, args, train=True, val=False, test=False, transform=None):
        """

        :param dataframe: If google, only google; If all, concat
        :param transform:
        """
        self.df = self.__load_dataset(args, train, val, test)
        self.transform = transform

    def __load_dataset(self, args, train, val, test):
        # info = LoadInfo()
        if train:
            info = LoadTrain()
            # train_content = info.loc[info['type'] == 'train']
            train_content = info
            if args.traindata == 'google':
                ans = train_content.loc[train_content['source'] == 'google']
            elif args.traindata == 'flickr':
                ans = train_content.loc[train_content['source'] == 'flickr']
            elif args.traindata == 'all':
                ans = train_content

        elif val:
            info = LoadVal()
            # ans = info.loc[info['type']=='val']
            ans = info
        elif test:
            info = LoadTest()
            # ans = info.loc[info['type']=='test']
            ans = info

        return ans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_line = self.df.iloc[idx, :]
        img_path = img_line.loc['image_path']
        img_label = float(img_line.loc['label'])
        # image = io.imread(img_path)
        image = Image.open(img_path)  # PIL.Image.Image对象
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
