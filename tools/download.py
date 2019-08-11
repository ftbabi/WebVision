import requests
from tqdm import tqdm
import logging
import os
import re
from threading import Thread


class DownloadBigFile:
    def __init__(self):
        pass


def support_continue(url):
    headers = {
        'Range': 'bytes=0-4'
    }
    try:
        r = requests.head(url, headers=headers)
        crange = r.headers['content-range']
        total = int(re.match(r'^bytes 0-4/(\d+)$', crange).group(1))
        return True, total
    except:
        pass
    try:
        total = int(r.headers['content-length'])
    except:
        total = 0
    return False, total

def test_pause(url_file, dir, id=0, block=512, headers = {}):
    # url_file = 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar'
    # url_file = 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt'
    filename = url_file[url_file.rfind('/') + 1:]
    filepath = os.path.join(dir, filename)
    flag, total = support_continue(url_file)
    selfsize = 0
    finished = False
    size = selfsize
    if flag:
        try:
            with open(filepath, 'rb') as f:
                while True:
                    cur = f.read(block)
                    if cur:
                        selfsize += block
                    else:
                        break
                # selfsize = int(f.read())
                size = selfsize + 1
        except FileNotFoundError:
            with open(filepath, 'w') as f:
                pass
        finally:
            headers['Range'] = "bytes=%d-" % (selfsize,)
    else:
        return -1
    r = requests.get(url_file, stream=True, verify=False, headers=headers)
    status_code = r.status_code
    if total > 0:
        print("[+] Size: %dKB" % (total / 1024))
    else:
        print("[+] Size: None")

    if status_code == 206:
        print("Continue to saving %s" % url_file)
        logging.info("Continue to saving %s" % url_file)
        with open(filepath, 'ab+') as f:
            f.seek(selfsize)
            f.truncate()
            try:
                for chunk in tqdm(r.iter_content(chunk_size=block)):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
                        f.flush()

                finished = True
            except:
                print("Download pause: %s" % url_file)
            finally:
                if not finished:
                    print("Not finished: %s" % url_file)
    else:
        print("Already finished: %s" % url_file)



def test(url_file, dir, id=0, block=512):
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
            for chunk in tqdm(r.iter_content(chunk_size=block)):
                if chunk:
                    f.write(chunk)
                    f.flush()
        print("Thread %d, Finish %s" % (id, filepath))
        logging.info("Thread %d, Finish %s" % (id, filepath))


class MultiThread(Thread):
    def __init__(self, worklist, outdir, id, headers={'Proxy-Connection': 'keep-alive'}):
        super().__init__()
        self.worklist = worklist
        self.id = id
        self.dir = outdir
        self.headers = headers
        # print("Thread "+str(id), cmdlist)

    def run(self):
        for url in self.worklist:
            # test(url, self.dir, self.id)
            test_pause(url, self.dir, self.id, 512, self.headers)
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
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_images_resized.tar',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/test_filelist.txt',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/val_images_resized.tar',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/val_filelist.txt',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/info.tar',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/google.tar',
        # 'https://data.vision.ee.ethz.ch/cvl/webvision2018/flickr.tar'
    ]
    for i in range(19, 24):
        image_url = 'https://data.vision.ee.ethz.ch/aeirikur/webvision2018/webvision_train_%02d' % i + '.tar'
        filelist.append(image_url)

    headers = {'Proxy-Connection': 'keep-alive'}
    # for url in filelist:
        # test(url, dir)
        # print(support_continue(url))
        # test_pause(url, dir, 0, 512, headers)

    logging.info("Begin download...")
    threadlst = []
    threadnum = 4
    for i in range(threadnum):
        # print(cmdlst[i::4])
        inputlst = filelist[i::threadnum]
        # print(inputlst)
        t = MultiThread(inputlst, dir, i)
        threadlst.append(t)
        # t.start()

    for thread in threadlst:
        thread.start()
