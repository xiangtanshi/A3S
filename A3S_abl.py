from Data_feature import *
from metric import *
from copy import deepcopy
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='The hyper-parameter setting for active probabilistic clustering.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=0, help='the data_path index')
parser.add_argument('--n', type=int, default=0, help='the N index')
parser.add_argument('--iso', type=int, default=0, help='the isotonic regressor index')
parser.add_argument('--types', type=int, default=0, help='0 represents features and 1 represents probability matrix')
parser.add_argument('--neighbor', type=int, default=50, help='the number of neighbors during FPC')
parser.add_argument('--T', type=int, default=2)
parser.add_argument('--clus', type=str, default='fpc', choices=['fpc','kmeans','spec','aggo'],help='the target clustering algorithm that BMAAS enhances with human query')
parser.add_argument('--tau', type=float, default=0.80, help='threshold for purity')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


start_time = time.time()
# --------------------------------------------------------------------------------------
# load data features or estimated posterior probability matrix and perform adaptive clustering
if args.types == 0:
    # load feature, then calculate the posterios probability matrix
    features,targetIds = get_feature_similarity(args.d, args.n, args.types)
    N = n_list[args.n]
    k = len(np.unique(targetIds))
    
    # record target clustering information
    target_dict = dict()
    for i in range(N):
        key = targetIds[i]
        if key not in target_dict:
            target_dict[key] = {i}
        else:
            target_dict[key].add(i)

    # perform fast probabilistic clustering to get adaptive cluster number K
    pclus_label, log_prob_dict, knn, cluster_dict = fast_prob_clustering(target_features=features,iso_reg_path=iso_paths[args.iso],n_neighbors=args.neighbor)
    print('Features loaded and finish the fast prob clustering.')
    # calculate the entire probability matrix (for the efficiency of latter operation, but is not necessary: we can still calculate a p_ij when required, and add a judge code)
    l2_dists_square = np.zeros((N,N))
    for i in range(N):
        delta = features[i] - features
        dist_i = np.sum(delta**2,axis=1)
        l2_dists_square[i] = dist_i
    iso_reg_model = joblib.load(iso_paths[args.iso])
    Prob_mat = iso_reg_model.predict(-l2_dists_square.flatten()).reshape(l2_dists_square.shape)
    Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)

elif args.types == 1:
    # directly load estimated probability matrix
    Prob_mat, targetIds = get_feature_similarity(args.d, args.n, args.types)
    N = n_list[args.n]
    k = len(np.unique(targetIds))

    # record target clustering information
    target_dict = dict()
    for i in range(N):
        key = targetIds[i]
        if key not in target_dict:
            target_dict[key] = {i}
        else:
            target_dict[key].add(i)
    # perform fast probabilistic clustering to get adaptive cluster number K
    pclus_label, log_prob_dict, knn, cluster_dict = fast_prob_clustering(Prob_mat=Prob_mat,types='matrix',n_neighbors=args.neighbor)

def dict_info(name='pred'):
    record = dict()
    if name == 'pred':
        for key in cluster_dict.keys():
            length = len(cluster_dict[key])
            if length not in record:
                record[length] = 1
            else:
                record[length] += 1
    elif name == 'real':
        for key in target_dict.keys():
            length = len(target_dict[key])
            if length not in record:
                record[length] = 1
            else:
                record[length] += 1
    record_list = [[item,record[item]] for item in record if item != 0]
    record_list.sort(key=lambda x: x[0])
    return record_list

info = dict_info(name='pred')
K = 0
for item in info:
    if item[0] != 1:
        K += item[1]


kmeans = KMeans(init="k-means++", n_clusters=K, n_init=4, random_state=0)
kmeans = kmeans.fit(features)
kmeans_label = kmeans.labels_

Affinity = deepcopy(Prob_mat)
# Affinity[Affinity<0.1] = 0
spec_label = spectral_clustering(Affinity, n_clusters=K, eigen_solver="arpack")

aggo = AgglomerativeClustering(n_clusters=K, linkage="average", metric="euclidean")
aggo.fit(features)
aggo_label = aggo.labels_

label_list = [pclus_label, kmeans_label, spec_label, aggo_label]
nmi_matrix = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        nmi_matrix[i,j] = ARI(label_list[i],label_list[j])
print(nmi_matrix)