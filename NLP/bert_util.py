from bert_serving.client import BertClient
from WebVision.dataloader import MetaLoader
from WebVision.dataloader import QuerySynsetMapLoader
import yaml
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import re
from tqdm import tqdm
from sklearn.cluster import KMeans



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
            count -= 1
            if count <= 0:
                break
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
            split_sent = re.split(pattern, sentence)
            result = title + '.'
            for sent in split_sent:
                sent = sent.strip()
                if sent and label.lower() in sent.lower():
                    sentence = sent + '.'
                    break

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


class TinyTest:
    def __init__(self, num, max_seq_len):
        self.num = num
        self.max_seq_len = max_seq_len

    def getNum(self):
        return self.num

    def getQueryDataEmb(self, savemeta, savelabel):
        querypath = '../data/2017/info/queries_google.txt'
        queryloader = MetaLoader(querypath)
        query = queryloader.getData()
        qhandler = FilterQuery()
        qres = np.array(qhandler.fit(query))

        metatest = ('../data/2017/meta/google/q%04d' % self.num) + '.json'
        metaloader = MetaLoader(metatest)
        meta = metaloader.getData()
        handler = FilterMeta()
        label_line = query[num-1]
        label = label_line.split(' ')[-1].strip()
        res = handler.fit(meta, label)

        np.savetxt(savemeta, res, fmt='%f', delimiter=',')
        np.savetxt(savelabel, qres, fmt='%f', delimiter=',')

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
            for i in range(n_clusters):
                out_buff[i] = []
                sqrt_buff.append([])
                summ.append(np.where(label_pred==i)[0].shape[0])


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

if __name__ == '__main__':
    bert_util_log = '../log/bert_util.log'

    logging.basicConfig(level=logging.INFO,  # 定义输出到文件的log级别，
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                        filename=bert_util_log,  # log文件名
                        filemode='a')
    test()
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
    scaler = 1
    normalize = '_word'
    metapath = './metainfo' + str(num) + '_128' + normalize + '.txt'
    labelpath = './labelinfo' + str(num) + '_128' + normalize + '.txt'
    drawpath = './draw' + str(num) + '_128' + normalize + '.txt'
#     # tttest.getQueryDataEmb(metapath, labelpath)
    queryemb, metaemb = tttest.loadQueryDataEmb(metapath, labelpath)
#     # dotpro = tttest.dotprod(queryemb[num-1, :], metaemb)
    dot = tttest.distance(queryemb[num-1, :], metaemb)
#     # np.savetxt(drawpath, dotpro, delimiter=',')
#
#     # dot = np.loadtxt(drawpath, delimiter=',')
    tttest.draw(dot, scaler)
#
    metatest = ('../data/2017/meta/google/q%04d' % num) + '.json'
    metaloader = MetaLoader(metatest)
    meta = metaloader.getData()
    # tttest.filter(meta, dot, 12, scaler)

# Visualize using tensorboard
    path = tttest.cluster_path(metaemb, dot, metainfo=meta, n_clusters=4)
    handler = FilterMeta()
    handler.visualize(metaemb, path)