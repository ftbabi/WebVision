import os
import re
import numpy as np
from nltk.parse import stanford
from WebVision.dataloader import MetaLoader
import nltk
from nltk.parse.stanford import StanfordDependencyParser
from tqdm import tqdm
import time
import logging


class SentenceParser:
    count = 0

    def __init__(self, num , model_path="/home/shaoyidi/VirtualenvProjects/myRA/"
                                  "WebVision/tools/stanfordparser/"
                                  "stanford-parser-full-2018-10-17/"
                                  "edu/stanford/nlp/models/lexparser/"
                                  "englishPCFG.ser.gz"):
        self.count += 1
        self.num = num
        if self.count == 1:
            os.environ[
                'STANFORD_PARSER'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser.jar'
            os.environ[
                'STANFORD_MODELS'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
        self.parser = stanford.StanfordParser(model_path=model_path)

    def getNum(self):
        return self.num

    def fit(self, metalist, label):
        '''

        :param metalist: all the meta info
        :param label: the class label
        :return: binary vector
        '''

        start = time.time()
        result = []
        count = 0
        for i in tqdm(metalist):
            count += 1
            title = i['title']
            desc = i['description']
            parse = self.__parse(title, desc, label)
            if parse:
                result.append(self.__valid(parse, label))
            else:
                result.append(False)

        end = time.time()
        logging.info("Time cost: %s" % str(end-start))

        return result

    def __parse(self, title, desc, label):
        '''
        Need filter signs

        :param title:
        :param desc:
        :param label:
        :return: sentence parse tree, nltk.tree
        '''
        # res = {'title':         None,
        #        'description':   None
        #        }
        #
        # if title:
        #     res['title'] = self.parser.parse(title.split())
        # if desc:
        #     res['description'] = self.parser.parse(desc.split())
        # sentence = title + ' ' + desc
        sentence = self.__getSentence(title, desc, label)
        if sentence:
            try:
                pattern = r',|\.|;|\?|!|，|。|、|；|‘|’|【|】|·|！|…|\||\]|\[|\(|\)'
                # pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
                # test_text = 'b,b.b/b;b\'b`b[b]b<b>b?b:b"b{b}b~b!b@b#b$b%b^b&b(b)b-b=b_b+b，b。b、b；b‘b’b【b】b·b！b b…b（b）b'
                split_sent = re.split(pattern, sentence)
                res = []
                for sent in split_sent:
                    sent = sent.strip()
                    if sent and label.lower() in sent.lower():
                        res.append(self.parser.parse(nltk.word_tokenize(sent)))
            except ValueError:
                print("Warning: ValueError: %s" % sentence)
                res = []
        else:
            res = []

        return res

    def transform(self, result):
        for order, i in enumerate(result):
            if not i:
                cand = meta[order]
                sentence = 'Rank:%s\tTitle: %s\tDesc: %s\tURL: %s' % (
                cand['rank'], cand['title'], cand['description'], cand['url'])
                logging.info(sentence)
                print(sentence)

    def __getSentence(self, title, desc, label):
        if label.lower() in title.lower():
            return title.lower()
        elif label.lower() in desc.lower():
            return desc.lower()
        else:
            return ""

    def __valid(self, parse, label):
        for sent in parse:
            while True:
                try:
                    each = next(sent)
                    for component in each.pos():
                        if label.lower() in component[0].lower():
                            if component[1] != 'JJ':
                                return True

                except StopIteration:
                    break

        return False

def getSymbol(meta):
    poollist = set()
    for i in meta:
        title = i['title']
        desc = i['description']
        sentence = title + desc
        for ch in sentence:
            if ch != ' ' and ch != '\n' and not ch.isalpha() and not ch.isdigit():
                poollist.add(ch)

    return poollist

def test():
    # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
    os.environ['STANFORD_PARSER'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,  # 定义输出到文件的log级别，
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                        filename='../log/nltk_util.log',  # log文件名
                        filemode='a')
    # logging.info("hello, it's me")
    # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
    # os.environ['STANFORD_PARSER'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser.jar'
    # os.environ['STANFORD_MODELS'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

    # 为JAVAHOME添加环境变量
    # java_path = "C:/Program Files (x86)/Java/jdk1.8.0_11/bin/java.exe"
    # os.environ['JAVAHOME'] = java_path

    # 句法标注
    # parser = stanford.StanfordParser(
    #     model_path="/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    # sentences = parser.parse("20 superb tench tips!".split())
    # tmp = next(sentences)
    # print(tmp.leaves())
    # print(tmp)
    # print(list(tmp.subtrees()))
    # print(tmp.pos())
    # for i in tmp.pos():
    #     if 'photo' in i:
    #         print(i)
    #         print(type(i))
    # print(tmp.leaf_treeposition(3))
    # print(tmp[0][1][1][0])

    # handler = SentenceParser()
    # res = handler.fit([{'title':'It\'s Tench', 'description':'Beautiful TEnch'}], 'tenCH')
    # print(res)

    # test = list(sentences)
    # for i in test:
    #     print(i)
    # print(test)

    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')

    # text = "Tench Aerial."
    # tokens = nltk.word_tokenize(text)
    # print(tokens)
    #
    # tags = nltk.pos_tag(tokens)
    # print(tags)

    # eng_parser = StanfordDependencyParser()
    # sentence = 'It\'s a tench island'
    # res = list(eng_parser.parse(sentence.split()))
    # print(res)
    # for row in res[0].triples():
    #     print(row)

    # meta = np.loadtxt('./metainfo.txt', delimiter=',')
    # label = np.loadtxt('./labelinfo.txt', delimiter=',')

    handler = SentenceParser(num=17)
    num = handler.getNum()
    metatest = ('../data/2017/meta/google/q%04d' % num) + '.json'
    metaloader = MetaLoader(metatest)
    meta = metaloader.getData()
    # meta = [meta[95]]


    # for i in meta:
    #     if i['rank'] == '729':
    #         handler.fit([i], 'tench')
    ans = handler.fit(meta, 'hen')
    handler.transform(ans)

    # poollist = getSymbol(meta)
    # print(poollist)

    # print(re.sub('[^\.,a-zA-Z0-9]', '', 'asd., -=3'))

