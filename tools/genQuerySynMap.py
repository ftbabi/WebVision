import os
import json
import pandas as pd
from devkit.webvision.config import LoadTrain


def genMap(outputpath):
    data = LoadTrain(percentage=100, select='google')
    ans = {}
    for index, row in data.iterrows():
        class_id = row['class_id']
        label = row['label']
        if not class_id in ans.keys():
            ans[class_id] = label

    with open(outputpath, 'w') as f:
        f.write(json.dumps(ans, ensure_ascii=False))

if __name__ == '__main__':
    genMap('/mnt/SSD/webvision/2017/info/queries_synset_map.json')