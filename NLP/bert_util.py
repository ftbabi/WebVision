import sys
sys.path.append("../")
from bert_serving.client import BertClient
from dataloader import MetaLoader
import yaml
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import re
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans

# DATA_BASE = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/data/2018/'
# DATA_BASE = '/home/ydshao/VirtualProjects/WebVision/data/2018'
DATA_BASE = '/mnt/SSD/webvision/2018'
INFO = os.path.join(DATA_BASE, 'info')
DATA_SOURCE = ['google', 'flickr']
META_FOLDER = os.path.join(DATA_BASE, 'meta')
EMB = os.path.join(DATA_BASE, 'emb')
DIS = os.path.join(DATA_BASE, 'distance')
QUERY_NUM = 12597
FILTER_GOOGLE = os.path.join(INFO, 'filter_google.txt')
FILTER_FLICKR = os.path.join(INFO, 'filter_flickr.txt')




def test():
    bc = BertClient()
    ans = bc.encode(['tench', 'Tench - Wikipedia', 'Tench Island', 'tench'])
    # ans = bc.encode(['King eat a pie', 'Queen eat a pie', 'Man eat a pie', 'Woman eat a pie'])
    # words = bc.encode(['King', 'Queen', 'Man', 'Woman'])
    # kq = np.dot(ans[0], ans[1].T)/(np.linalg.norm(ans[0]) * np.linalg.norm(ans[1]))
    # kmwq = np.dot(ans[0]-ans[2]+ans[3], ans[1].T) / (np.linalg.norm(ans[0]-ans[2]+ans[3]) * np.linalg.norm(ans[1]))
    # dis = np.linalg.norm(ans[0]-ans[1])
    # disq = np.linalg.norm(ans[0]-ans[2]+ans[3] - ans[1])
    # ans = words
    # kqw = np.dot(ans[0], ans[1].T)/(np.linalg.norm(ans[0]) * np.linalg.norm(ans[1]))
    # kmwqw = np.dot(ans[0]-ans[2]+ans[3], ans[1].T) / (np.linalg.norm(ans[0]-ans[2]+ans[3]) * np.linalg.norm(ans[1]))
    # disw = np.linalg.norm(ans[0] - ans[1])
    # disqw = np.linalg.norm(ans[0] - ans[2] + ans[3] - ans[1])

    test = np.linalg.norm(np.array([3, 4]))

    norm_0 = np.linalg.norm(ans[0])
    norm_1 = np.linalg.norm(ans[1])
    norm_2 = np.linalg.norm(ans[2])

    ss = np.dot(ans[0], ans[1].T)
    sss = ss / (norm_0*norm_1)
    dd = np.dot(ans[0], ans[2].T)
    ddd = dd / (norm_0*norm_2)
    ll = np.dot(ans[0], ans[0].T)
    lll = ll / (norm_0*norm_0)

    dissim = np.linalg.norm(ans[1] - ans[0])
    disdiff = np.linalg.norm(ans[2] -ans[0])

    norm03 = np.linalg.norm(ans[0]-ans[3])
    norm13 = np.linalg.norm(ans[1]-ans[3])
    norm23 = np.linalg.norm(ans[2]-ans[3])
    sim = np.dot(ans[0] - ans[3], (ans[1] - ans[3]).T)
    diff = np.dot((ans[0]-ans[3]), (ans[2]-ans[3]).T)
    slf = np.dot(ans[0]-ans[3], (ans[0]-ans[3]).T)

    norm_sim = sim/(norm03*norm13)
    norm_diff = diff/(norm03*norm23)
    slf = slf/(norm03*norm03)
    print(ans)

def testdraw():
    input1 = tf.constant([1., 2., 3.], name='input1')
    input2 = tf.Variable([tf.random_uniform([200]) for i in range(3)], name='input2')
    # output = input1 + input2

    writer = tf.summary.FileWriter('../log')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'test'

    projector.visualize_embeddings(writer, config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.save(sess, '../log/test.log', 1)
    writer.close()


def train(metaarray):
    x = tf.placeholder(tf.float32, metaarray.shape)
    # W = tf.Variable(tf.ones())
    # tfpoints = tf.Variable(np.array(metaarray), name='points')
    bias = tf.Variable(tf.ones(metaarray.shape), name='bias')
    tfpoints = x+bias

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(tfpoints, feed_dict={x:metaarray})

    return res

class FilterQuery:
    def __init__(self, config_path='../config.yaml'):
        # Load configuration
        f = open(config_path)
        content = yaml.load(f)
        f.close()
        self.relation = int(content['relation'])
        self.normalize = int(content['normalize'])
        self.take_title = int(content['title'])
        self.take_desc = int(content['desc'])

        # Initial bert
        self.bc = BertClient()

    def fit(self, metalist):
        result = []
        count = 20
        for i in tqdm(metalist):
            # count -= 1
            # if count <= 0:
            #     break
            line = i.split(' ')
            query = line[-1].strip()
            emb = self.__emb(query)
            # if self.relation:
            #     query = "It's " + query
            #     emb_r = self.__emb(query)
            #     emb = emb_r - emb
            # Normalize
            if self.normalize:
                emb = self.__normalize(emb)
            result.extend(emb)

        return result

    def __normalize(self, emb):
        emb = emb / np.linalg.norm(emb)
        return emb

    def __emb(self, query):
        emb = self.bc.encode([query])
        return emb


class FilterMeta:
    def __init__(self, config_path='../config.yaml'):
        # Load configuration
        f = open(config_path)
        content = yaml.load(f)
        f.close()
        self.relation = int(content['relation'])
        self.normalize = int(content['normalize'])
        self.take_title = int(content['title'])
        self.take_desc = int(content['desc'])

        # Initial bert
        self.bc = BertClient()

    def fit(self, metalist, label=''):
        result = []
        for i in tqdm(metalist):
            # if int(i['rank']) < 208:
            #     continue
            title = i['title']
            desc = i['description']
            sentence = self.__getSentence(title, desc, label)
            emb = self.__emb(sentence)
            # if self.relation:
            #     label = label.split(' ')
            #     label = label[-1].strip()
            #     labelemb = self.__emb(label)
            #     emb = emb - labelemb
            # Normalize
            if self.normalize:
                emb = self.__normalize(emb)
            result.extend(emb)

        return result

    def __normalize(self, emb):
        # Original data has been normalized
        emb = emb / np.linalg.norm(emb)
        # norm = np.linalg.norm(emb)
        return emb

    def __selected(self, piece, label):
        lpattern = r'\+| '
        # label = 'abc def'
        label_split = re.split(lpattern, label)
        for pi in label_split:
            if pi and pi.lower() in piece.lower():
                return True

        return False

    def __getSentence(self, title, desc, label):
        sentence = ""
        if self.take_title:
            # if label.lower() in title.lower():
            sentence += title + '. '
        if self.take_desc:
            # if label.lower() in desc.lower():
            sentence += desc

        if not sentence:
            sentence = title + '. ' + desc
            pattern = r',|\.|;|\?|!|，|。|、|；|‘|’|【|】|·|！|…|\||\]|\[|\(|\)'
            # pattern = r',|\.|;|\?|!|，|。|、|；|‘|’|【|】|·|！|…|\||\]|\[|\(|\)'
            split_sent = re.split(pattern, sentence)
            cand = []
            for sent in split_sent:
                sent = sent.strip()
                if sent and self.__selected(sent, label):
                    cand.append(sent)
                    # sentence = sent + '.'
                    # break
            if len(cand) > 0:
                sentence = ','.join(cand) + '.'

        return sentence

    def __emb(self, sentence):
        emb = self.bc.encode([sentence])
        return emb

    def visualize(self, points, meta_path='', log_path='/home/shaoyidi/VirtualenvProjects/myRA/WebVision/log'):
        # debut = np.array(points)
        tfpoints = tf.Variable(np.array(points), name='tfpoints')

        writer = tf.summary.FileWriter(log_path)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = tfpoints.name

        if meta_path:
            embedding.metadata_path = meta_path

        projector.visualize_embeddings(writer, config)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.save(sess, '../log/test.log')
        writer.close()


class BertHandler:
    def __init__(self, num, max_seq_len):
        self.num = num
        self.max_seq_len = max_seq_len

    def genQueryEmb(self):
        # Init
        queryfile = 'queries.txt'
        querypath = os.path.join(INFO, queryfile)

        # Read and Emb
        queryloader = MetaLoader(querypath)
        query = queryloader.getData()
        qhandler = FilterQuery()
        qres = np.array(qhandler.fit(query))

        # Save
        savepath = os.path.join(EMB, queryfile)
        np.savetxt(savepath, qres, fmt='%f', delimiter=',')

    def genMetaEmb(self):
        pass
        # Init
        # metatest = ('../data/2017/meta/google/q%04d' % self.num) + '.json'
        # metaloader = MetaLoader(metatest)
        # meta = metaloader.getData()
        # handler = FilterMeta()
        # label_line = query[num - 1]
        # label = label_line.split(' ')[-1].strip()
        # res = handler.fit(meta, label)

    def loadQueryEmb(self):
        pass

    def loadMetaEmb(self):
        pass

    def filter(self):
        pass


class TinyTest:
    def __init__(self, num, max_seq_len):
        self.num = num
        self.max_seq_len = max_seq_len

    def getNum(self):
        return self.num

    def __getLabel(self, labelline):
        label_idx = labelline.find(' ')
        label = labelline[label_idx+1:]

        return label.strip()

    def getQueryDataEmb(self, queryfile='queries.txt'):
        print("Get embeddings")
        # Init Queries
        querypath = os.path.join(INFO, queryfile)

        # Read and Emb
        print("Load queries from %s" % querypath)
        logging.info("Load queries from %s" % querypath)
        queryloader = MetaLoader(querypath)
        query = queryloader.getData()
        qhandler = FilterQuery()
        print("Embed queries")
        logging.info("Embed queries")
        qres = np.array(qhandler.fit(query))
        print("Finish queries emb")
        logging.info("Finish queries emb")

        # Save Query
        savequery = os.path.join(EMB, queryfile)
        np.savetxt(savequery, qres, fmt='%f', delimiter=',')
        print("Queries emb save to %s" % savequery)
        logging.info("Queries emb save to %s" % savequery)

        print("Process meta")
        logging.info("Process meta")
        for dataset in DATA_SOURCE:
            dataset_folder = os.path.join(META_FOLDER, dataset)
            for i in range(1, QUERY_NUM+1):
                # Init
                filename = 'q%05d.json' % i
                metapath = os.path.join(dataset_folder, filename)

                # Read and Emb
                print("==> Load meta from %s" % metapath)
                logging.info("==> Load meta from %s" % metapath)
                metaloader = MetaLoader(metapath)
                meta = metaloader.getData()
                handler = FilterMeta()
                label_line = query[i-1]
                # label = label_line.split(' ')[-1].strip()
                label = self.__getLabel(label_line)
                print("==> Embed meta")
                logging.info("==> Embed meta")
                res = handler.fit(meta, label)
                print("==> Finish meta emb")
                logging.info("==> Finish meta emb")

                # Save meta
                savemeta = os.path.join(EMB, dataset)
                savemeta = os.path.join(savemeta, 'q%05d.txt'%i)
                np.savetxt(savemeta, res, fmt='%f', delimiter=',')
                print("==> Meta emb saved to %s" % savemeta)
                logging.info("==> Meta emb saved to %s" % savemeta)

            # metatest = ('../data/2017/meta/google/q%04d' % self.num) + '.json'
            # metaloader = MetaLoader(metatest)
            # meta = metaloader.getData()
            # handler = FilterMeta()
            # label_line = query[num-1]
            # label = label_line.split(' ')[-1].strip()
            # res = handler.fit(meta, label)
        print("Finish embedding")
        logging.info("Finish embedding")

    def computeDistance(self, queryfile='queries.txt'):
        # Load Query
        loadquery = os.path.join(EMB, queryfile)
        print("Load queries emb from %s" % loadquery)
        logging.info("Load queries emb from %s" % loadquery)
        queryemb = np.loadtxt(loadquery, delimiter=',')

        print("Load meta emb")
        logging.info("Load meta emb")
        for dataset in DATA_SOURCE:
            for i in range(1, QUERY_NUM + 1):
                loadmeta = os.path.join(EMB, dataset)
                loadmeta = os.path.join(loadmeta, 'q%05d.txt' % i)
                print("==> Load meta emb from %s" % loadmeta)
                logging.info("==> Load meta emb from %s" % loadmeta)
                metaemb = np.loadtxt(loadmeta, delimiter=',')

                print("==> Compute distance")
                logging.info("==> Compute distance")
                distance = self.distance(queryemb[i-1, :], metaemb)

                # Save distance
                savedis = os.path.join(DIS, dataset)
                savedis = os.path.join(savedis, 'q%05d.txt' % i)
                np.savetxt(savedis, distance, fmt='%f', delimiter=',')
                print("==> Distance saved to %s" % savedis)
                logging.info("==> Distanced save to %s" % savedis)

        print("Finish distance")
        logging.info("Finish distance")

    def filterFormal(self, queryfile='queries.txt', threshold=0.6):
        # Load Query
        # loadquery = os.path.join(EMB, queryfile)
        # print("Load queries emb from %s" % loadquery)
        # logging.info("Load queries emb from %s" % loadquery)
        # queryemb = np.loadtxt(loadquery, delimiter=',')

        print("Load meta emb")
        logging.info("Load meta emb")
        for dataset in DATA_SOURCE:
            filter_list_path = os.path.join(INFO, "filter_%s.txt"%dataset)
            total = 0
            with open(filter_list_path, 'w') as f:
                for i in range(1, QUERY_NUM + 1):
                    # Load emb
                    loadmeta = os.path.join(EMB, dataset)
                    loadmeta = os.path.join(loadmeta, 'q%05d.txt' % i)
                    print("==> Load meta emb from %s" % loadmeta)
                    logging.info("==> Load meta emb from %s" % loadmeta)
                    metaemb = np.loadtxt(loadmeta, delimiter=',')

                    # Load distance
                    loaddistance = os.path.join(DIS, dataset)
                    loaddistance = os.path.join(loaddistance, 'q%05d.txt' % i)
                    distance = np.loadtxt(loaddistance, delimiter=',')

                    # Cluster
                    print("==> Cluster")
                    logging.info("==> Cluster")
                    cluster_idx = self.__cluster(metaemb)

                    # Get threshold and save ans
                    clean_idx = self.__getCleanSet(distance, cluster_idx, threshold)
                    squeeze_clean = np.hstack(clean_idx)
                    ans = np.zeros(metaemb.shape[0], dtype=int)
                    ans[squeeze_clean] = 1
                    f.write('\n'.join([str(x) for x in ans]))
                    f.write('\n')
                    print("==> Saved part %d" % i)
                    logging.info("==> Saved part %d" % i)
                    total += metaemb.shape[0]
            print("%s total: %d" % (dataset, total))
            logging.info("%s total: %d" % (dataset, total))

        print("Finish filter")
        logging.info("Finish filter")

    def loadQueryDataEmb(self, savemeta, savelabel):
        metaemb = np.loadtxt(savemeta, delimiter=',')
        queryemb = np.loadtxt(savelabel, delimiter=',')

        return queryemb, metaemb

    def draw(self, dataemb, scaler=1):
        dot = dataemb
        data = {}
        for i in dot:
            try:
                order = int(i*scaler)
            except ValueError:
                continue
            if order in data.keys():
                data[order] += 1
            else:
                data[order] = 1
        for key, val in data.items():
            print("Score: %d, Num: %3d" % (key, val))
        plt.bar(data.keys(), data.values())
        plt.show()

    def dotprod(self, label, meta):
        # ans = np.dot(queryemb[num - 1, :], metaemb.T)
        ans = np.dot(label, meta.T)
        return ans

    def distance(self, label, meta):
        ans = []
        for i in meta:
            dis = np.linalg.norm(i-label)
            ans.append(dis)
        return np.array(ans)

    def filter(self, meta, dot, thresh=60, scaler=1):
        count = 0
        total = 0
        selected = []
        ind_scr = {}
        ind = 0

        for order, i in enumerate(dot):
            total += 1
            if i > thresh/scaler:
                count += 1
                cand = meta[order]
                selected.append(cand)
                ind_scr[ind] = i
                ind += 1

        sorted_cand = sorted(ind_scr.items(), key=lambda x: x[1])
        for cand_idx in sorted_cand:
            cand = selected[int(cand_idx[0])]
            i = float(cand_idx[1])
            print("Score: %f\tRank: %5s\tTitle: %s\tDescription: %s\tURL: %s" % (i, cand['rank'], cand['title'], cand['description'], cand['url']))
            logging.info("Num: %d\tMaxSeqLen: %d\tThresh: %f\tScore: %f\tRank: %s\tTitle: %s\tDescription: %s\tURL: %s" %
                             (self.num, self.max_seq_len, thresh, i, cand['rank'], cand['title'], cand['description'], cand['url']))

        print("Filtered files: %d/%d" % (count, total))
        logging.info("Filtered files: %d/%d" % (count, total))

    def __cluster(self, meta, n_clusters=5):
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit(meta)
        label_pred = estimator.labels_
        selected = []

        for i in range(n_clusters):
            # sqrt_buff.append([])
            selected.append(np.where(label_pred == i)[0])

        return selected

    def __getCleanSet(self, score, cluster_id, threshold):
        cluster_score = {}
        for order, cluster in enumerate(cluster_id):
            cur_score = np.array(score[cluster])
            means = np.mean(cur_score)
            var = np.var(cur_score)
            cluster_score[order] = means

        # Rise
        sorted_cluster = sorted(cluster_score.items(), key=lambda x: x[1])
        total = score.shape[0]
        summary = 0
        result = []
        for cell in sorted_cluster:
            group_num = cell[0]
            summary += len(cluster_id[group_num])
            result.append(cluster_id[group_num])
            if summary >= total * threshold:
                break

        return result


    def cluster_path(self, meta, score, metainfo=[], n_clusters=3,
                     outputpath='/home/shaoyidi/VirtualenvProjects/myRA/WebVision/NLP/cluster.tsv',
                     showoff=30):
        estimator = KMeans(n_clusters=n_clusters)
        estimator.fit(meta)
        label_pred = estimator.labels_
        with open(outputpath, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(label_pred):
                f.write("%d\t%d\n" % (index, label))

        if metainfo:
            out_buff = {}
            sqrt_buff = []
            summ = []
            selected = []
            for i in range(n_clusters):
                out_buff[i] = []
                sqrt_buff.append([])
                selected.append(np.where(label_pred==i)[0])
                summ.append(np.where(label_pred==i)[0].shape[0])

            self.__getCleanSet(score, selected, 0.4)

            for order, pred in enumerate(label_pred):
                if len(out_buff[pred]) <= showoff:
                    out_buff[pred].append(order)

            for key, vals in out_buff.items():
                for idx in vals:
                    cand = metainfo[idx]
                    print("Cluster: %d\tScore: %f\tRank: %5s\tTitle: %s\tDescription: %s\tURL: %s" % (
                            key, score[idx], cand['rank'], cand['title'], cand['description'], cand['url']))
                    # else:
                    #     print("Cluster: %d\tRank: %s\tTitle: %s\tDescription: %s\tURL: %s" % (
                    #         key, cand['rank'], cand['title'], cand['description'], cand['url']))
                    sqrt_buff[key].append(score[idx])

            for order, sqrt_single in enumerate(sqrt_buff):
                var = np.var(np.array(sqrt_single))
                means = np.mean(np.array(sqrt_single))
                # print("Cluster %d, Num: %d" % (i, summ[i]))
                print("Cluster: %d, Num: %3d, mean: %f, var: %f" % (order, summ[order], means, var))

        return outputpath

def informal_use():
    # test()
    # testdraw()

    # num = 1
    #
    # querypath = '../data/2017/info/queries_google.txt'
    # queryloader = MetaLoader(querypath)
    # query = queryloader.getData()
    # qhandler = FilterQuery()
    # qres = np.array(qhandler.fit(query))
    #
    # metatest = '../data/2017/meta/google/q000' + str(num) + '.json'
    # metaloader = MetaLoader(metatest)
    # meta = metaloader.getData()
    # handler = FilterMeta()
    # res = handler.fit(meta)
    # ans = train(np.array(res))

    # ans = np.array(res)
    # handler.visualize(ans)
    # print(res)

    # dotpro = np.dot(qres[num-1, :], ans.T)

    # Filter
    # thresh = 12.5/100
    # thresh = 60
    # for order, num in enumerate(dotpro):
    #     if num < thresh:
    #         cand = meta[order]
    #         print("Title: %s\tDesc: %s\tUrl: %s" % (cand['title'], cand['description'], cand['url']))

    # Draw
    # xaxis = []
    # for i, order in enumerate(dotpro):
    #     xaxis.append(order)
    # plt.scatter(xaxis, list(np.sort(dotpro)))
    # plt.show()

    # Save
    # with open('draw.txt', 'w') as f:
    #     for i in np.sort(dotpro):
    #         f.write(str(i))
    #         f.write('\n')

    # emb
    tttest = TinyTest(num=1, max_seq_len=128)
    num = tttest.getNum()
    # tttest.getQueryDataEmb(metapath, labelpath)

    scaler = 1
    normalize = '_word'
    metapath = './metainfo' + str(num) + '_128' + normalize + '.txt'
    labelpath = './labelinfo' + str(num) + '_128' + normalize + '.txt'
    # folder = os.path.join(EMB, 'google')
    # metapath = os.path.join(folder, 'q%05d.txt' % num)
    # labelpath = os.path.join(EMB, 'queries.txt')
    drawpath = './draw' + str(num) + '_128' + normalize + '.txt'

    # Load emb
    queryemb, metaemb = tttest.loadQueryDataEmb(metapath, labelpath)
    #     # dotpro = tttest.dotprod(queryemb[num-1, :], metaemb)
    dot = tttest.distance(queryemb[num - 1, :], metaemb)
    #     # np.savetxt(drawpath, dotpro, delimiter=',')
    #
    #     # dot = np.loadtxt(drawpath, delimiter=',')
    tttest.draw(dot, scaler)
    #
    # metatest = os.path.join(folder, 'q%05d.txt' % num)
    metatest = '../data/2017/meta/google/q%04d.json' % num
    metaloader = MetaLoader(metatest)
    meta = metaloader.getData()
    # tttest.filter(meta, dot, 12, scaler)

    # Visualize using tensorboard
    path = tttest.cluster_path(metaemb, dot, metainfo=meta, n_clusters=4)
    #     handler = FilterMeta()
    #     handler.visualize(metaemb, path)

    a = np.array([0,1,0,1,0,0,1,5, 6, 5, 6])
    s1 = np.where(a==1)[0]
    s5 = np.where(a==5)[0]
    s6 = np.where(a==6)[0]
    a_lst = []
    a_lst.append(s1)
    a_lst.append(s5)
    a_lst.append(s6)
    selected = np.hstack(a_lst)

    b = np.zeros(a.shape[0], dtype=int)
    b[selected] = 2
    test_str = '-'.join([str(x) for x in b])
    print(test_str)


def run():
    parser = argparse.ArgumentParser(description='Training on WebVision',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--traindata', type=str, choices=['google', 'flickr', 'all'],
    #                     help='Choose among google/flickr/all.', default='all')
    parser.add_argument('--genemb', action='store_true', help='Generate the embeddings', default=False)
    parser.add_argument('--filter', action='store_true', help='Filter the embeddings', default=False)
    parser.add_argument('--compute', action='store_true', help='Compute distance', default=False)
    # parser.add_argument('--cluster', action='store_true', help='Cluster', default=False)
    parser.add_argument('--maxseqlen', '-l', type=int, default=128, help='Max sequeence length')
    # Optimization options
    # parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train.')
    # parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
    # parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    # parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    # parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    # # parser.add_argument('--test_bs', type=int, default=10)
    # # parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
    # #                     help='Decrease learning rate at these epochs.')
    # # parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    #
    # # Checkpoints
    # parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    # parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
    # # parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    #
    # # Architecture
    # parser.add_argument('--classes', type=int, default=5000, help='Number of classes.')
    # parser.add_argument('--model', type=str, choices=['resnext50_32x4d', 'resnext101_32x8d', 'densenet121'],
    #                     help='Choose among resnext50_32x4d/densenet121.', default='resnext101_32x8d')
    #
    # # parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    # # parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    # # parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    # # parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    #
    # # Acceleration
    # parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    # # parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    #
    # # i/o
    # parser.add_argument('--log', type=str, default='./', help='Log folder.')
    #
    # # Choices
    # parser.add_argument('--train', action='store_true', help='train the model', default=False)
    # parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set', default=False)
    # parser.add_argument('--predict', action='store_true',
    #                     help='predict the answers for test set with trained model', default=False)

    # Parse
    args = parser.parse_args()

    # args.genemb = True

    handler = TinyTest(1, args.maxseqlen)
    if args.genemb:
        handler.getQueryDataEmb()
    if args.compute:
        handler.computeDistance()
    if args.filter:
        handler.filterFormal()


if __name__ == '__main__':
    bert_util_log = '/home/ydshao/VirtualProjects/WebVision/log/bert_util.log'
    if not os.path.exists(bert_util_log):
        print("%s not exists" % bert_util_log)
        exit(-1)

    logging.basicConfig(level=logging.INFO,  # 定义输出到文件的log级别，
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                        filename=bert_util_log,  # log文件名
                        filemode='a')

    # informal_use()
    run()