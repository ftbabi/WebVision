from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
from nltk.parse import stanford
import os
from WebVision.dataloader import MetaLoader
from tqdm import tqdm
import nltk
from WebVision.NLP.nltk_util import SentenceParser
import re

def example():
    nlp = StanfordCoreNLP('/home/shaoyidi/VirtualenvProjects/myRA/WebVision/NLP/model/stanford-corenlp-full-2018-10-05')
    # 这里改成你stanford-corenlp所在的目录
    sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    print('Tokenize:', nlp.word_tokenize(sentence))
    print('Part of Speech:', nlp.pos_tag(sentence))
    print('Named Entities:', nlp.ner(sentence))
    print('Constituency Parsing:', nlp.parse(sentence))
    print('Dependency Parsing:', nlp.dependency_parse(sentence))

    nlp.close()

def test(sentence):
    # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
    os.environ['STANFORD_PARSER'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
    parser = stanford.StanfordParser(
        model_path="/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sentences = parser.parse(nltk.word_tokenize(sentence.lower()))
    tmp = next(sentences)
    print(tmp.leaves())
    print(tmp)

    handler = SentenceParser(1)
    res = handler.fit([{'title':sentence, 'description':''}], 'crawdad')
    print(res)

if __name__ == '__main__':
    # sentence = 'Tench island'
    # nlp = StanfordCoreNLP('/home/shaoyidi/VirtualenvProjects/myRA/WebVision/NLP/model/stanford-corenlp-full-2018-10-05')
    # # 这里改成你stanford-corenlp所在的目录
    # os.environ[
    #     'STANFORD_PARSER'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser.jar'
    # os.environ[
    #     'STANFORD_MODELS'] = '/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
    #
    # parser = stanford.StanfordParser(
    #     model_path="/home/shaoyidi/VirtualenvProjects/myRA/WebVision/tools/stanfordparser/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    # sentences = parser.parse(nltk.word_tokenize(sentence))
    # tmp = next(sentences)
    # # ans =  nlp.parse(sentence)
    # # tree = Tree.fromstring(ans)
    # # print(type(ans))
    # # print(tree)
    # print(tmp)
    # print(sentence.split())
    # text = "Tench Aerial."
    # tokens = nltk.word_tokenize(sentence)
    # print(tokens)

    # print('hE'.lower() in "HelLo".lower())
    # num = 1
    # metatest = '../data/2017/meta/google/q000' + str(num) + '.json'
    # metaloader = MetaLoader(metatest)
    # meta = metaloader.getData()
    # count = 0
    # for i in tqdm(meta):
    #     title = i['title']

    sentence = "Crawdad"
    # pattern = r',|\.|;|\?|!|，|。|、|；|‘|’|【|】|·|！|…|\||\]|\[|\(|\)'
    # pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    # test_text = 'b,b.b/b;b\'b`b[b]b<b>b?b:b"b{b}b~b!b@b#b$b%b^b&b(b)b-b=b_b+b，b。b、b；b‘b’b【b】b·b！b b…b（b）b'
    # split_sent = re.split(pattern, sentence)
    # print(split_sent)
    # for st in split_sent:
    #     test(st)
    test(sentence)