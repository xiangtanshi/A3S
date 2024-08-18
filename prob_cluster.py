import os
import json
import base64
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans, spectral_clustering, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
import random


def fast_prob_clustering(target_features=None, Prob_mat=None, types='vector', epsilon=0.0001, n_neighbors=50, iso_reg_path=None):
    # perform fast prob_clustering on single-view data (features) or multi-view data (aggregated probability matrix)

    # step 1: get the knn neighbors list and the log_prob_dict
    if types == 'vector':
        N = target_features.shape[0]
        neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(target_features)
        dist, knn = neigh.kneighbors(target_features)
        row_col, prob, contained_row_col_set = [], [], set()
        iso_reg_model = joblib.load(iso_reg_path)
        batch_prob = iso_reg_model.predict(-(dist**2).flatten()).reshape(dist.shape)
        # batch_prob[np.isnan(batch_prob)] = 0.0
        for i in range(N):
            for index, j in enumerate(knn[i]):
                key = (min(i,j),max(i,j))
                if key in contained_row_col_set:
                    continue
                row_col.append(key)
                contained_row_col_set.add(key)
                prob.append(batch_prob[i][index])
        prob = np.clip(np.asarray(prob,dtype=float),epsilon,1-epsilon)
        log_prob = np.log(prob/(1-prob))
    elif types == 'matrix':
        N = Prob_mat.shape[0]
        prob = np.clip(Prob_mat,epsilon,1-epsilon)
        log_prob = np.log(prob/(1-prob))
        log_prob = log_prob.flatten()
        row_col = []
        for i in range(N):
            for j in range(N):
                row_col.append((i,j))
        # construct the knn based on the Prob_mat
        knn = np.zeros((N,n_neighbors))
        for i in range(N):
            p = prob[i]
            ind = np.argsort(p)[::-1]
            knn[i] = ind[:n_neighbors]
        knn = knn.astype(int)

    # step 2: construct the cluster_dict and preparation structure&function
    log_prob_dict = dict(zip(row_col, log_prob))
    partition, cluster_dict = np.arange(N), dict([(i,{i}) for i in range(N)])

    def get_point_set_likelihood(i, cluster_id):
        cluster = cluster_dict[cluster_id]
        likelihood = 0.0
        for j in cluster:
            if i == j:
                continue
            key = (min(i,j),max(i,j))
            if key not in log_prob_dict:
                l2_dist_square = np.sum((target_features[i]-target_features[j])**2)
                p_ij = iso_reg_model.predict(np.array([-l2_dist_square]))
                p_ij = min(1-epsilon, max(epsilon, p_ij))
                log_prob_dict[key] = np.log(p_ij/(1-p_ij))
            likelihood += log_prob_dict[key]
        return likelihood

    # step 3: perform the greedy probability clustering
    update = True
    while update:
        update = False
        for i in range(N):
            candidate_clusters = set(partition[knn[i]])
            cur_cluster_likelihood = get_point_set_likelihood(i, partition[i])
            max_cluster_id, max_cluster_likelihood = partition[i], 0.0
            for cluster_id in candidate_clusters:
                if cluster_id == partition[i]:
                    continue
                likelihood = get_point_set_likelihood(i, cluster_id) - cur_cluster_likelihood
                if likelihood > max_cluster_likelihood:
                    max_cluster_id = cluster_id
                    max_cluster_likelihood = likelihood
            if partition[i] != max_cluster_id:
                update = True
                cluster_dict[max_cluster_id].add(i)
                cluster_dict[partition[i]].remove(i)
                partition[i] = max_cluster_id

    # reorganize the labels and cluster_dict
    _, par = np.unique(partition, False, True)
    classes = len(np.unique(par))
    cluster_dict = dict()
    for i in range(classes):
        cluster_dict[i] = set()
    for i in range(N):
        cluster_dict[par[i]].add(i)
    return par, log_prob_dict, knn, cluster_dict

