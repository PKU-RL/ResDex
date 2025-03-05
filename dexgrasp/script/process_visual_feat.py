import torch
from sklearn.cluster import KMeans
import numpy as np

import os
os.environ["OMP_NUM_THREADS"] = "9"

NUM_CLUSTERS = 5
# load the visual_feat_data.pt
data_dir = "Intermediate_Data/"
visual_feat_data = torch.load(data_dir+'visual_feat_data.pt')
id2scale = torch.load(data_dir+'id2scale.pt')
object_code_list = torch.load(data_dir+'object_code_list.pt')
object_scale_id_list = torch.load(data_dir+'object_scale_id_list.pt')


# visual_feat_data is a dictionary with 2 keys
# I want to perfrom k-means on the visual features, and get the keys 
# for the k-means clusters centers
# I will use the k-means++ algorithm


# get the visual features
visual_index = []
visual_features = []
for key1, value1 in visual_feat_data.items():
    for key2, value2 in value1.items():
        visual_features.append(value2)
        visual_index.append((key1, key2, value2))


#convert visual feature to cpu
visual_features = [x.cpu().numpy() for x in visual_features]
visual_features = np.array(visual_features)


# perform k-means
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(visual_features)


# find the most similar object to the k-means cluster centers, one object per cluster
index = []
for center in kmeans.cluster_centers_:
    min_dist = float('inf')
    min_index = 0
    for i in range(len(visual_features)):
        dist = np.linalg.norm(center - visual_features[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    index.append((visual_index[min_index][0], visual_index[min_index][1]))

# print the object name and scale for the most similar object to the k-means cluster centers
for i in range(len(index)):
    print(object_code_list[index[i][0]],id2scale[index[i][1]])

# for every cluster, sort the objects by the distance to the cluster center, from the closest to the farthest
# save them to 5 different files
# create a file folder named NUM_CLUSTERS
# save the files to the folder
import os
if not os.path.exists(str(NUM_CLUSTERS)):
    os.makedirs(str(NUM_CLUSTERS))


for i in range(NUM_CLUSTERS):
    index = []
    for j in range(len(visual_features)):
        if kmeans.labels_[j] == i:
            index.append(j)
    index = sorted(index, key=lambda x: np.linalg.norm(kmeans.cluster_centers_[i] - visual_features[x]))
    with open("clusters/"+str(NUM_CLUSTERS)+'/cluster_'+str(i)+'.txt', 'w') as f:
        for j in index:
            f.write(object_code_list[visual_index[j][0]] + ' ' + str(id2scale[visual_index[j][1]]) + '\n')
