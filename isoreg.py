import os
import json
import base64
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import faiss

def faiss_search_approx_knn(query, target, k):
    # for market-1501 datasets with normalized features in a sphere space
    cpu_index = faiss.IndexFlatIP(target.shape[1])

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = False
    co.usePrecomputed = False
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=1)
    try:
        gpu_index.add(target)
    except:
        raise ValueError('cannot load feature to GPU')
    dists, nbrs = gpu_index.search(query, k=k)
    del gpu_index

    return dists, nbrs

def get_feature_from_file(resource_file=None):
    # get the features and targetIds of the market-1501 datasets
    features, targetIds = [], []
    with tqdm(os.path.getsize(resource_file)) as pbar:
        f = open(resource_file, "r")
        for line in f:
            pbar.update(len(line))
            obj = json.loads(line)
            featureString, targetId = obj['features'], obj['targetId']
            features.append(np.frombuffer(base64.b64decode(featureString), dtype=np.float32))
            targetIds.append(int(targetId[-4:])-1)   # transform the string to numbers (start from 0)
        f.close()
    features, targetIds = np.asarray(features), np.array(targetIds)
    return features, targetIds

def compute_posterior_prob_by_l2_distance(resource_file=None, model_path=None, Features=None, K=2, n_neighbors=100, f_type='file'):
    # learn a mapping function: -l2_distance^2 -> pairwise probability with the isotonic regression
    # Features: (n_samples, n_features), K: number of ground_truth classes
    if f_type == 'file':
        features, targetIds = get_feature_from_file(resource_file)
        dists, nbrs = faiss_search_approx_knn(features, features, n_neighbors)
        X = 2 * dists.flatten() - 2

    elif f_type == 'vector':
        features = Features
        neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
        dists, nbrs = neigh.kneighbors(features)
        dists = dists**2
        X = - dists.flatten()

    if model_path:
        #get psuedo label
        kmeans_clus = KMeans(n_clusters=K,init='k-means++',n_init='auto').fit(features)
        # labels = kmeans_clus.labels_
        labels = targetIds
        y = np.asarray(labels[nbrs.flatten()] == labels[nbrs[:, [0]*n_neighbors]].flatten(), dtype=int)
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(X, y)
        # save the mapping function
        joblib.dump(iso_reg, model_path, compress=9)
        return 1
    else:
        raise ValueError('no path to store the isotonic regression model!')
    
if __name__ == '__main__':
     
    k = 100
    compute_posterior_prob_by_l2_distance(resource_file='data/datasets/market{}.txt'.format(k),
                                          model_path='data/models/gt_market_iso_reg_{}.pkl'.format(k), K=k, n_neighbors=80)  