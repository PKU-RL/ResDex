import yaml
import numpy as np

a = yaml.safe_load(open("cfg/new_train_set.yaml", "r"))
a = list(a.items())

b = a[0]

scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }
c = np.load("../assets/pcldata/"+b[0]+scale2str[b[1][0]]+".npy")

print(c.shape)
# print(max([ len(value) for key, value in a.items()]))