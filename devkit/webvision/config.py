# Configuration file for the data folders.

from __future__ import print_function
import os
import json
import pandas as pd
import re
from os.path import join, isdir, basename, isfile
from glob import glob

# Global configuration
# DATA_BASE = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/data/2018/'
# DATA_BASE = '/home/ydshao/VirtualProjects/WebVision/data/2018/'
DATA_BASE = '/mnt/SSD/webvision/2017/'
INFO = join(DATA_BASE, 'info')
DATA_SOURCE = ['google', 'flickr']
TRAIN_FOLDER = join(DATA_BASE, 'train_images_256')
VAL_FOLDER = join(DATA_BASE, 'val_images_256')
VAL_DIVIDE_FOLDER = join(DATA_BASE, 'val_images_divide')
TEST_FOLDER = join(DATA_BASE, 'test_images_256')
META_FOLDER = join(DATA_BASE, 'meta')
CACHE_FOLDER = join(DATA_BASE, 'cache')
# CLASS_NUM = 12597 # 2018 version
CLASS_NUM = 2204 # 2017 version


def _ExtractClassName(image_id):
    if re.match('.*/q\d{5}/.*', image_id):
        return int(image_id.split('/')[-2][1:])
    else:
        return -1


def _ParseTextFile(filename, columns=None):
    data = pd.read_csv(filename,
                       delim_whitespace=True,
                       header=None)
    if columns:
        data.columns = columns
    return data


def LoadInfo():
    '''
    Load the dataset info from the dataset
    '''
    print("Loading configuration from %s" % (INFO))
    # Load train file list
    all_train_data = []
    for dataset in DATA_SOURCE:
        # Load training file
        trn = _ParseTextFile(
            join(INFO, 'train_filelist_%s.txt' % (dataset)),
            ['image_id', 'label'])
        print('Data %s has %d training samples' % (dataset, trn.shape[0]))
        trn['type'] = 'train'
        trn['source'] = dataset
        # Load meta
        meta = _ParseTextFile(join(INFO, 'train_meta_list_%s.txt' % (dataset)),
                              ['meta_path', 'row_number'])
        meta['meta_path'] = meta.meta_path.map(lambda x: join(META_FOLDER, x))
        all_train_data.append(trn.join(meta[['meta_path']]))
    training_df = pd.concat(all_train_data)
    training_df['image_path'] = training_df['image_id'].map(
        lambda x: join(TRAIN_FOLDER, x))
    # Load testing file list
    test_df = _ParseTextFile(
        join(INFO, 'test_filelist.txt'),
        ['image_id'])
    test_df['type'] = 'test'
    test_df['image_path'] = test_df['image_id'].map(
        lambda x: join(TEST_FOLDER, x))
    # Load validation file list
    val_df = _ParseTextFile(
        join(INFO, 'val_filelist.txt'),
        ['image_id', 'label'])
    val_df['type'] = 'val'
    val_df['image_path'] = val_df['image_id'].map(
        lambda x: join(VAL_FOLDER, x))
    data_info = pd.concat([training_df, val_df, test_df])
    data_info['class_id'] = \
        data_info['image_id'].map(_ExtractClassName)
    return data_info


def ValidateIntegrity(info):
    # Validate all files exist
    for _, row in info.iterrows():
        assert isfile(
            row.image_path), 'Image file does not exist %s' % row.image_path
        if row.type == 'train':
            assert isfile(
                row.meta_path), 'Meta file does not exist %s' % row.meta_path

    return True

def LoadTrain(argsfilter=False, select='all', percentage=0, random=False):
    '''
        Load the dataset info from the dataset
        '''
    if not os.path.exists(CACHE_FOLDER):
        os.mkdir(CACHE_FOLDER)

    print("Loading configuration from %s" % (INFO))
    # Load train file list

    cur_datasource = []
    if select == 'all':
        # Load All
        if not random:
            cache_file = os.path.join(CACHE_FOLDER, 'train_filelist_all_%d.json' % percentage)
        else:
            cache_file = os.path.join(CACHE_FOLDER, 'train_filelist_all_%d_random.json' % percentage)
        if os.path.exists(cache_file):
            # Load from cache
            print("Load from cache %s" % cache_file)
            data_info = pd.read_json(cache_file)
        else:
            cur_datasource = DATA_SOURCE
            data_info = LoadTrainAndSave(argsfilter, cur_datasource, cache_file, percentage, random)
    else:
        if not random:
            cache_file = os.path.join(CACHE_FOLDER, 'train_filelist_%s_%d.txt' % (select, percentage))
        else:
            cache_file = os.path.join(CACHE_FOLDER, 'train_filelist_%s_%d_random.txt' % (select, percentage))
        if os.path.exists(cache_file):
            print("Load from cache %s" % cache_file)
            data_info = pd.read_json(cache_file)
        else:
            cur_datasource.append(select)
            data_info = LoadTrainAndSave(argsfilter, cur_datasource, cache_file, percentage, random)

    return data_info

def LoadTest():
    # Load testing file list
    test_df = _ParseTextFile(
        join(INFO, 'test_filelist.txt'),
        ['image_id'])
    test_df['type'] = 'test'
    test_df['image_path'] = test_df['image_id'].map(
        lambda x: join(TEST_FOLDER, x))

    data_info = test_df
    data_info['class_id'] = \
        data_info['image_id'].map(_ExtractClassName)
    return data_info

def LoadVal():
    # Load validation file list
    val_df = _ParseTextFile(
        join(INFO, 'val_filelist.txt'),
        ['image_id', 'label'])
    val_df['type'] = 'val'
    val_df['image_path'] = val_df['image_id'].map(
        lambda x: join(VAL_FOLDER, x))
    data_info = val_df
    data_info['class_id'] = \
        data_info['image_id'].map(_ExtractClassName)
    return data_info

def LoadTrainAndSave(argsfilter, src_lst, cache_file, percentage, random):
    # Load from original files
    print("Load training data from disk")
    all_train_data = []
    for dataset in src_lst:
        # Load training file
        trn = _ParseTextFile(
            join(INFO, 'train_filelist_%s.txt' % (dataset)),
            ['image_id', 'label'])
        print('Data %s has %d training samples' % (dataset, trn.shape[0]))
        trn['type'] = 'train'
        trn['source'] = dataset
        # Load meta
        meta = _ParseTextFile(join(INFO, 'train_meta_list_%s.txt' % (dataset)),
                              ['meta_path', 'row_number'])
        print("Mapping meta_path")
        meta['meta_path'] = meta.meta_path.map(lambda x: join(META_FOLDER, x))

        if argsfilter:
            # Load filter
            filterinfo = _ParseTextFile(
                join(INFO, 'filter_%s.txt' % (dataset)),
                ['selected']
            )

            all_train_data.append(trn.join([meta[['meta_path']], filterinfo['selected']]))
        else:
            all_train_data.append(trn.join(meta[['meta_path']]))

    training_df = pd.concat(all_train_data)
    print("Mapping image_path")
    training_df['image_path'] = training_df['image_id'].map(
        lambda x: join(TRAIN_FOLDER, x))

    data_info = training_df
    print("Mapping class_id")
    data_info['class_id'] = \
        data_info['image_id'].map(_ExtractClassName)

    print("Select data according to args.subset: %d" % percentage)
    if percentage == 100:
        final_data_info = data_info
    else:
        percent_data = []
        for i in range(1, CLASS_NUM+1):
            cur_data = data_info.loc[data_info['class_id'] == i]
            length = len(cur_data)
            selected_num = int(length * percentage / 100)
            print("Selecting data with class id: %5d, number of selected: %3d/%3d" % (i, selected_num, length))
            if not random:
                selected_data = cur_data.iloc[0:selected_num]
            else:
                selected_data = pd.DataFrame.sample(cur_data, n=selected_num)
            percent_data.append(selected_data)
        final_data_info = pd.concat(percent_data)

    print("Save to cache %s" % cache_file)
    final_data_info = final_data_info.reset_index()
    final_data_info.to_json(cache_file)

    return final_data_info

def LoadQuerySynsetMap():
    filename = os.path.join(INFO, 'queries_synsets_map.json')
    with open(filename, 'r') as f:
        load_dict = json.load(f)

    return load_dict

if __name__ == '__main__':
    info = LoadInfo()
    train_set = info.loc[info['type']=='train']

    print("Hello")