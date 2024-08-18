from Data_feature import *
from metric import *
from copy import deepcopy
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, NPU

parser = argparse.ArgumentParser(description='The hyper-parameter setting for active probabilistic clustering.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=0, help='the data_path index')
parser.add_argument('--n', type=int, default=0, help='the N index')
parser.add_argument('--method', type=str, default='random', choices=['random','ffqs','npu'])
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

# load the data
features,targetIds = get_feature_similarity(args.d, args.n)
N = n_list[args.n]
k = len(np.unique(targetIds))
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

def Random_P(query_threshold_list):
    
    start_time = time.time()
    upper = query_threshold_list[-1]
    absolue_index = np.random.choice([p for p in range(N*N)],size=2*upper,replace=True).tolist()
    pair_set = []
    for i in range(2*upper):
        item = absolue_index.pop(0)
        row = int(item/N)
        col = item % N
        if row != col and [row,col] not in pair_set:
            pair_set.append([row,col])
    ml, cl = [], []
    q = 0
    checkpoint = query_threshold_list.pop(0)
    while q < upper and len(pair_set):
        pair = pair_set.pop(0)
        if state[pair[0],pair[1]] == 0:
            q += 1
        if query(targetIds,pair[0],pair[1]):
            state[pair[0],pair[1]] = 1
        else:
            state[pair[0],pair[1]] = -1
        state_resolution(pair)
        if q == checkpoint:
            clusterer = PCKMeans(n_clusters=k, max_iter=50)
            clusterer.fit(features,ml=ml,cl=cl)
            label = clusterer.labels_
            nmi,ari = test(targetIds,label) 
            nmi_record.append(nmi)
            ari_record.append(ari)
            if len(query_threshold_list):
                checkpoint = query_threshold_list.pop(0)
            else:
                break
    result = [checkpoint_list_,nmi_record,ari_record]
    print(result)

    end_time = time.time()
    print('total running time of Random: {}seconds.'.format(end_time-start_time))

def FFQS(query_threshold_list):
    start_time = time.time()
    for threshold in query_threshold_list:
        oracle = ExampleOracle(targetIds,max_queries_cnt=threshold)
        active_learner = ExploreConsolidate(n_clusters=k)
        active_learner.fit(features,oracle=oracle)
        constraints = active_learner.pairwise_constraints_
        clusterer = PCKMeans(n_clusters=k, max_iter=100)
        clusterer.fit(features,ml=constraints[0],cl=constraints[1])
        nmi,ari = test(targetIds,clusterer.labels_)
        nmi_record.append(nmi)
        ari_record.append(ari)
    end_time = time.time()
    result = [checkpoint_list_,nmi_record,ari_record]
    print(result)
    print('total running time of FFQS: {}seconds.'.format(end_time-start_time))

def NPU_(query_threshold_list):
    start_time = time.time()
    for threshold in query_threshold_list:
        oracle = ExampleOracle(targetIds,max_queries_cnt=threshold)
        clusterer = PCKMeans(n_clusters=k, max_iter=100)
        active_learner = NPU(clusterer=clusterer)
        active_learner.fit(features,oracle=oracle)
        constraints = active_learner.pairwise_constraints_
        clusterer = PCKMeans(n_clusters=k, max_iter=100)
        clusterer.fit(features,ml=constraints[0],cl=constraints[1])
        nmi,ari = test(targetIds,clusterer.labels_)
        nmi_record.append(nmi)
        ari_record.append(ari)
    end_time = time.time()
    result = [checkpoint_list_,nmi_record,ari_record]
    print(result)
    print('total running time of FFQS: {}seconds.'.format(end_time-start_time))


checkpoint_list = [1] + [int(4000/6*i) for i in range(1,7)]
checkpoint_list_ = deepcopy(checkpoint_list)
if args.method == 'random':
    Random_P(checkpoint_list)
elif args.method == 'ffqs':
    FFQS(checkpoint_list)
elif args.method == 'npu':
    NPU_(checkpoint_list)