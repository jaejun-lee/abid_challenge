import os
from os import listdir
import os.path
import numpy as np
import random
import json
import pandas as pd 

img_dir= "/home/jaejun/workspace/abid_challenge/dataset/data/bin-images/"
meta_dir = "/home/jaejun/workspace/abid_challenge/dataset/data/metadata/"

def get_quantity(d):
    quantity = d['EXPECTED_QUANTITY']
    return quantity

def make_counting_list(img_dir,meta_dir, limit = 5):

    lst_count = []
    img_list = listdir(img_dir)
    N = len(img_list)
    #N = 535234
    for i in range(N):
        if i%1000 == 0:
            print("get_metadata: processing (%d/%d)..." % (i,N))
        jpg_path = '%s%05d.jpg' % (img_dir,i+1)
        jpg_name = '%05d.jpg' % (i+1)
        json_path = '%s%05d.json' % (meta_dir,i+1)

        if os.path.isfile(jpg_path) and os.path.isfile(json_path):
            d = json.loads(open(json_path).read())
            quantity = get_quantity(d)
            if quantity <= limit:
                lst_count.append([jpg_name, quantity])

    print("get_metadata: Available Images: %d" % len(lst_count))
    return lst_count

def make_counting_df(img_dir,meta_dir, limit = 5):
    lst_count = make_counting_list(img_dir,meta_dir, 7)
    df = pd.DataFrame(lst_count, columns=["id", "label"])
    df['label'] = df['label'].apply(str)
    return df

if __name__ == "__main__":

    df = make_counting_df(img_dir,meta_dir, 7)
    