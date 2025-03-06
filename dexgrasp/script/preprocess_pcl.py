
import numpy as np
import os, sys
# add ../ to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.util import get_pointcloud_from_mesh
import yaml

def process_object_list(object_list):
    scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }
    for key, value in object_list.items():
        for scale in value:
            keys = key.split('/')
            save_dir = "../assets/pcldata/"+keys[0]+'/'
            save_fn = keys[1]+scale2str[scale]+".npy"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            asset_root="../assets/meshdatav3_scaled/"
            fn = key + "/coacd/decomposed_" + scale2str[scale] +".obj"
            pc = get_pointcloud_from_mesh(asset_root, fn)
            np.save(save_dir+save_fn, pc)

if __name__ =="__main__":

    train_set=yaml.load(open("cfg/new_train_set.yaml", "r"), Loader=yaml.FullLoader)
    test_seen_cat=yaml.load(open("cfg/new_seen_cat.yaml", "r"), Loader=yaml.FullLoader)
    test_unseen_cat=yaml.load(open("cfg/new_unseen_cat.yaml", "r"),Loader=yaml.FullLoader)

   

    process_object_list(train_set)
    print("get train set")
    process_object_list(test_seen_cat)
    print("get seen cat")
    process_object_list(test_unseen_cat)
    print("get unseen cat")
