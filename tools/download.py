import requests
from tqdm import tqdm
import logging
import os
from threading import Thread


def test(url_file, dir, id=0):
    # url_file = 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar'
    # url_file = 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt'
    filename = url_file[url_file.rfind('/') + 1:]
    filepath = os.path.join(dir, filename)
    if os.path.exists(filepath):
        print("Already downloaded: %s" % filepath)
    else:
        print("Thread: %d, try downloading %s" % (id, url_file))
        logging.info("Thread: %d, try downloading %s" % (id, url_file))
        headers = {'Proxy-Connection': 'keep-alive'}
        r = requests.get(url_file, stream=True, headers=headers)
        # length = float(r.headers['content-length'])
        # print(length)
        print("Thread %d, begin to save to %s" % (id, filepath))
        logging.info("Thread %d, begin to save to %s" % (id, filepath))
        with open(filepath, 'wb') as f:
        # f = open("file_path", "wb")
            # chunk是指定每次写入的大小，每次只写了512byte
            for chunk in tqdm(r.iter_content(chunk_size=512)):
                if chunk:
                    f.write(chunk)
        print("Thread %d, Finish %s" % (id, filepath))
        logging.info("Thread %d, Finish %s" % (id, filepath))

class MultiThread(Thread):
    def __init__(self, worklist, outdir, id):
        super().__init__()
        self.worklist = worklist
        self.id = id
        self.dir = outdir
        # print("Thread "+str(id), cmdlist)

    def run(self):
        for url in self.worklist:
            test(url, self.dir, self.id)
        print("Thread %d finished" % self.id)

if __name__ == '__main__':
    logpath = '../log/download.log'
    dir = '../data/2018/download'
    if not os.path.exists(logpath):
        with open(logpath, 'w') as f:
            f.write('')
    logging.basicConfig(level=logging.INFO,  # 定义输出到文件的log级别，
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                        filename=logpath,  # log文件名
                        filemode='a')
    filelist = [
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/val_images_resized.tar',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/val_filelist.txt',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/info.tar',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/google.tar',
        'https://data.vision.ee.ethz.ch/cvl/webvision2018/flickr.tar'
    ]
    for i in range(33):
        image_url = 'https://data.vision.ee.ethz.ch/aeirikur/webvision2018/webvision_train_%02d' % i+'.tar'
        filelist.append(image_url)

    # for url in filelist:
    #     test(url, dir)

    logging.info("Begin download...")
    threadlst = []
    threadnum = 8
    for i in range(threadnum):
        #print(cmdlst[i::4])
        inputlst = filelist[i::threadnum]
        #print(inputlst)
        t = MultiThread(inputlst, dir, i)
        threadlst.append(t)
        #t.start()

    for thread in threadlst:
        thread.start()

