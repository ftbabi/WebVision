import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from devkit.webvision.config import _ParseTextFile, _ExtractClassName
from devkit.webvision.config import INFO

def split_by_first_space_second(x):
    idx = x.find(' ')
    order = x[:idx]
    query = x[idx+1:]
    return query

def split_by_first_space_first(x):
    idx = x.find(' ')
    order = x[:idx]
    query = x[idx+1:]
    return order

def getQueryMap(base_dir, train_prefix='../../2018/train_images_256', meta_prefix='../../2018/meta'):
    def getQuerySynMap(filepath):
        ans = {}
        with open(filepath, 'r') as f:
            query_syn_lst = f.readlines()
        for line in query_syn_lst:
            idx = line.find(' ')
            query = int(line[:idx])
            synset = int(line[idx+1:-1])
            if synset in ans.keys():
                ans[synset].append(query)
            else:
                ans[synset] = [query]

        return ans

    def mapAndSave(src, dst):
        pass

    src_qry = os.path.join(base_dir, '2017/info/queries_google.txt')
    tar_qry = os.path.join(base_dir, '2018/info/queries.txt')
    src_syn = os.path.join(base_dir, '2017/info/synsets.txt')
    tar_syn = os.path.join(base_dir, '2018/info/synsets.txt')
    src_query_syn = os.path.join(base_dir, '2017/info/queries_synsets_map.txt')
    tar_query_syn = os.path.join(base_dir, '2018/info/queries_synsets_map.txt')
    save = os.path.join(base_dir, '2017/map2018.txt')
    syn_syn_map = os.path.join(base_dir, '2017/syn_syn.txt')
    gen_query = os.path.join(base_dir, '2017/info/queries.txt')
    data_source = ['google', 'flickr']
    # gen_train_google = os.path.join(base_dir, '2017/info/train_filelist_google.txt')
    # gen_train_flickr = os.path.join(base_dir, '2017/info/train_filelist_flickr.txt')

    with open(src_qry, 'r') as f:
        src_qry_lst = f.readlines()
        src_qry_lst = list(map(split_by_first_space_second, src_qry_lst))
    with open(tar_qry, 'r') as f:
        tar_qry_lst = f.readlines()
        tar_qry_lst = list(map(split_by_first_space_second, tar_qry_lst))
    with open(src_syn, 'r') as f:
        src_syn_lst = f.readlines()
        src_syn_lst = list(map(split_by_first_space_first, src_syn_lst))
    with open(tar_syn, 'r') as f:
        tar_syn_lst = f.readlines()
        tar_syn_lst = list(map(split_by_first_space_first, tar_syn_lst))
    tar_qs_map = getQuerySynMap(tar_query_syn)
    src_qs_map = getQuerySynMap(src_query_syn)

    # with open(src, 'r') as f:
    #     src_lst = f.readlines()
    # with open(tar, 'r') as f:
    #     tar_lst = f.readlines()
    # src_lst = list(map(split_by_first_space_second(), src_lst))
    # tar_lst = list(map(split_by_first_space_second(), tar_lst))
    # src_pd = pd.read_csv(src, sep=' ', header=None)
    # tar_pd = pd.read_csv(tar, sep=' ', header=None)
    # tar_lst = tar_pd['1'].to_list()

    ans_index = []
    ans_map = []
    syn_syn_dict = {}
    for index, code in enumerate(src_syn_lst):
        # print(query)
        if not code in tar_syn_lst:
            print("Warning: Lack query %d, %s" % (index, code))
        else:
            if tar_syn_lst.index(code) in syn_syn_dict.keys():
                print("Warning: Dup syn %s" % (code))
            else:
                syn_syn_dict[tar_syn_lst.index(code)] = index
            tar_queries = tar_qs_map[tar_syn_lst.index(code)]
            src_queries = src_qs_map[index+1]
            print("Find map from %s,%s to %s,%s" % (str(src_queries), str([src_qry_lst[i-1] for i in src_queries]), str(tar_queries), str([tar_qry_lst[i-1] for i in tar_queries])))
            ans_index.extend(tar_queries)
            line = ','.join(list(map(lambda x: str(x), src_queries))) + '\t' + ','.join(list(map(lambda x: str(x), tar_queries)))
            ans_map.append(line)
        # tar_query = tar_pd.loc[tar_pd['1'] == query]
        # print(tar_query)
    # print(src_lst)
    with open(save, 'w') as f:
        f.write('\n'.join(ans_map))
    with open(syn_syn_map, 'w') as f:
        for key, val in syn_syn_dict.items():
            f.write(str(key) + ' ' + str(val))
            f.write('\n')

    with open(gen_query, 'w') as f:
        for i in ans_index:
            f.write(str(i))
            f.write(' ')
            f.write(tar_qry_lst[i-1])

    tar_train = LoadTrain(data_source)
    # Gen meta file list
    # Gen train file list
    for dataset in data_source:
        print("Process dataset %s" % (dataset))
        trainbuff = []
        metabuff = []
        gen_train_path = os.path.join(base_dir, '2017/info/train_filelist_%s.txt' % dataset)
        gen_meta_path = os.path.join(base_dir, '2017/info/train_meta_list_%s.txt' % dataset)
        cur_set = tar_train.loc[tar_train['source'] == dataset]
        for i in tqdm(ans_index):
            cand = cur_set.loc[cur_set['class_id'] == i]
            for index, row in cand.iterrows():
                train_line = os.path.join(train_prefix, row['image_id']) + ' ' + str(syn_syn_dict[row['label']])
                meta_line = os.path.join(meta_prefix, row['meta_path']) + ' ' + str(row['row_number'])
                trainbuff.append(train_line)
                metabuff.append(meta_line)

        print("Writing to files...")
        with open(gen_train_path, 'w') as f:
            f.write('\n'.join(trainbuff))
        with open(gen_meta_path, 'w') as f:
            f.write('\n'.join(metabuff))

def LoadTrain(DATA_SOURCE):
    '''
    Load the dataset info from the dataset
    '''
    print("Loading configuration from %s" % (INFO))
    # Load train file list
    all_train_data = []
    for dataset in DATA_SOURCE:
        # Load training file
        trn = _ParseTextFile(
            os.path.join(INFO, 'train_filelist_%s.txt' % (dataset)),
            ['image_id', 'label'])
        print('Data %s has %d training samples' % (dataset, trn.shape[0]))
        trn['type'] = 'train'
        trn['source'] = dataset
        # Load meta
        meta = _ParseTextFile(os.path.join(INFO, 'train_meta_list_%s.txt' % (dataset)),
                              ['meta_path', 'row_number'])
        # meta['meta_path'] = meta.meta_path.map(lambda x: os.path.join(META_FOLDER, x))
        all_train_data.append(trn.join(meta))
    training_df = pd.concat(all_train_data)

    data_info = training_df
    data_info['class_id'] = \
        data_info['image_id'].map(_ExtractClassName)
    return data_info

def test():
    path = '/mnt/SSD/webvision/2018/info'
    tar = '../../2017/info/synsets.txt'
    test_path = os.path.join(path, tar)
    with open(test_path, 'r') as f:
        content = f.readlines()
    print(content)

def modify_class(base_dir):
    data_source = ['google', 'flickr']
    for dataset in data_source:
        filepath = os.path.join(base_dir, '2017/info/train_filelist_%s.txt' % dataset)
        with open(filepath, 'r') as f:
            info = f.readlines()
        preclass = -1
        count = -1
        writebuff = []
        for line in tqdm(info):
            split_line = line.split(' ')
            if int(split_line[1]) != preclass:
                preclass = int(split_line[1])
                count += 1
            writebuff.append(split_line[0] + ' ' + str(count))
        with open(filepath, 'w') as f:
            f.write('\n'.join(writebuff))


if __name__ == '__main__':
    base_dir = '/mnt/SSD/webvision'
    getQueryMap(base_dir)
    # modify_class(base_dir)
    # test()