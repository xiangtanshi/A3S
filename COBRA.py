from Data_feature import *
from metric import *
from copy import deepcopy
import argparse
import time
import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='The hyper-parameter setting for active probabilistic clustering.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=2, help='the data_path index')
parser.add_argument('--n', type=int, default=2, help='the N index')
parser.add_argument('--K', type=int, default=86, help='the super-instance number')
parser.add_argument('--step', type=int, default=60, help='the test frequency')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

start_time = time.time()
# load the data
features,targetIds = get_feature_similarity(args.d, args.n)
N = n_list[args.n]
k = len(np.unique(targetIds))
K = args.K
state = np.eye(N)
ml,cl=[],[]
nmi_record = []
ari_record = []

def test(gt_label, pred_label):
    nmi=NMI(gt_label, pred_label)
    ari=ARI(gt_label, pred_label)
    purity=Purity(gt_label, pred_label)
    nmi,ari,purity = round(nmi,4), round(ari,4), round(purity,4)
    print(nmi,ari,purity)
    return nmi, ari

def query(gt_label, x, y):
    return gt_label[x] == gt_label[y]

def state_resolution(pair):
    # Fast transitivity inference: 
    # ensure that there are no pair (i,j) where S[i,j]=0, while S[i,j] could be induced by known chains such as <S[i,k1], S[k1,k2], S[k2,j]>
    for idx in pair:
        link_set = np.where(state[idx]==1)[0]
        sep_set = np.where(state[idx]==-1)[0]
        if len(link_set)>2:
            for p in link_set:
                for q in link_set:
                    if p != q:
                        state[p,q] = 1
                        if (p,q) not in ml:
                            ml.append((p,q))
                            ml.append((q,p))
        if len(sep_set) and len(link_set):
            for p in link_set:
                for q in sep_set:
                    state[p,q] = -1
                    state[q,p] = -1
                    if (p,q) not in cl:
                        cl.append((p,q))
                        cl.append((q,p))

# get initial clustering
kmeans = KMeans(init="k-means++", n_clusters=K, n_init=4, random_state=0)
kmeans = kmeans.fit(features)
label = kmeans.labels_
centroids = kmeans.cluster_centers_
center_indices = []
test(targetIds,label)


# calculate the euclidean distance
l2_dists_square = np.zeros((N,N))
for i in range(N):
    delta = features[i] - features
    dist_i = np.sum(delta**2,axis=1)
    l2_dists_square[i] = dist_i

cluster_dict = dict()
for i in range(K):
    cluster_dict[i] = set()
for i in range(N):
    cluster_dict[label[i]].add(i)

leader_dict = dict()
for key in cluster_dict:
    if len(cluster_dict[key]):
        leader_dict[key] = list(cluster_dict[key])[0]

def min_dist(key1,key2):
    set1 = list(cluster_dict[key1])
    set2 = list(cluster_dict[key2])
    dists = []
    for i in set1:
        for j in set2:
            dists.append(l2_dists_square[i,j])
    return min(dists)


def merge_class(key1,key2):
    set1 = list(cluster_dict[key1])
    for s1 in set1:
        label[s1] = label[leader_dict[key2]]
        cluster_dict[key2].add(s1)
    leader_dict.pop(key1)
    cluster_dict.pop(key1)

merged = 1
count = 0
checkpoint_list = [args.step*i for i in range(10)]
checkpoint_list_ = deepcopy(checkpoint_list)
checkpoint = checkpoint_list.pop(0)
stop = 0
while merged and not stop:
    merged = 0
    key_list = []
    for key in cluster_dict.keys():
        if len(cluster_dict[key]):
            key_list.append(key)

    P_list = []
    for i in range(len(key_list)-1):
        for j in range(i+1, len(key_list)):
            if state[leader_dict[key_list[i]],leader_dict[key_list[j]]] == -1:
                continue
            else:
                P_list.append([key_list[i],key_list[j],min_dist(key_list[i],key_list[j])])
    P_list = sorted(P_list, key=lambda x: x[2])
    while len(P_list):
        key1,key2,_ = P_list.pop(0)
        if key1 not in leader_dict or key2 not in leader_dict:
            continue
        s1,s2 = leader_dict[key1],leader_dict[key2]
        # if state[s1,s2] == 0:
        #     count += 1
        count += 1
        if query(targetIds,s1,s2):
            state[s1,s2] = 1
            merge_class(key1,key2)
            merged = 1
        else:
            state[s1,s2] = -1
        # state_resolution([s1,s2])

        if count > checkpoint:
            nmi,ari = test(targetIds,label)
            nmi_record.append(nmi)
            ari_record.append(ari)
            if len(checkpoint_list):
                checkpoint = checkpoint_list.pop(0)
            else:
                stop = 1
                break
result = [checkpoint_list_,nmi_record,ari_record]
print(result)
end_time = time.time()
print('total running time of COBRA: {}seconds.'.format(end_time-start_time))
print(len(leader_dict))