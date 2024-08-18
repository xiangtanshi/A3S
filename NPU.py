from Data_feature import *
from metric import *
from copy import deepcopy
import argparse
import time
import warnings
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='The hyper-parameter setting for active probabilistic clustering.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=0, help='the data_path index')
parser.add_argument('--n', type=int, default=0, help='the N index')
parser.add_argument('--iso', type=int, default=0, help='the isotonic regressor index')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


def test(gt_label, pred_label):
    nmi=NMI(gt_label, pred_label)
    ari=ARI(gt_label, pred_label)
    purity=Purity(gt_label, pred_label)
    nmi,ari,purity = round(nmi,4), round(ari,4), round(purity,4)
    print(nmi,ari,purity)
    return nmi, ari

def query(gt_label, x, y):
    return gt_label[x] == gt_label[y]


def NPU(query_threshold_list, batchsize=5):
    # perform Normalized Point-based Uncertainty (NPU) method
    start_time = time.time()

    # load the data
    features,targetIds = get_feature_similarity(args.d, args.n)
    N = n_list[args.n]
    k = len(np.unique(targetIds))
    state = np.eye(N)
    ml,cl=[],[]
    nmi_record = []
    ari_record = []
    # calculate the Probability Matrix to replace the random forest part in the original NPU, which aims to estimate a similarity matrix

    l2_dists_square = np.zeros((N,N))
    for i in range(N):
        delta = features[i] - features
        dist_i = np.sum(delta**2,axis=1)
        l2_dists_square[i] = dist_i
    iso_reg_model = joblib.load(iso_paths[args.iso])
    Prob_mat = iso_reg_model.predict(-l2_dists_square.flatten()).reshape(l2_dists_square.shape)
    Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)

    # define the relative function
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

    certain_neighbor_dict = dict()
    l_list = []
    un_list = [i for i in range(N)]
    count = 0
    # get initial clustering result
    clusterer = PCKMeans(n_clusters=k,max_iter=50)
    clusterer.fit(features,ml=[],cl=[])
    kmeans_label = clusterer.labels_
    nmi,ari = test(targetIds,kmeans_label)
    nmi_record.append(nmi)
    ari_record.append(ari)

    checkpoint = query_threshold_list.pop(0)
    # initialize the certain-set, make sure there are at least two neighborhoods are chosen
    init_sample = int(np.random.randint(0,N,1))
    certain_neighbor_dict[int(targetIds[init_sample])] = {init_sample}
    l_list.append(init_sample)
    un_list.remove(init_sample)
    while(len(certain_neighbor_dict)==1):
        sample = int(np.random.randint(0,N,1))
        count += 1
        if query(targetIds,init_sample,sample) == 1:
            state[init_sample,sample] = 1
            certain_neighbor_dict[int(targetIds[init_sample])].add(sample)
        else:
            state[init_sample,sample] = -1
            certain_neighbor_dict[int(targetIds[sample])] = {sample}
        state_resolution([init_sample,sample])
        l_list.append(sample)
        un_list.remove(sample)
    
    # start selecting the most informative sample batch by batch
    while count <= checkpoint:
        exp_num = np.zeros(len(un_list))
        l = len(certain_neighbor_dict)
        p = np.zeros((len(un_list),l))
        length = np.zeros(l)
        ind = 0
        key_order = []
        for key in certain_neighbor_dict:
            key_order.append(key)
            neigh = list(certain_neighbor_dict[key])
            length[ind] = len(neigh)
            for i,idx in enumerate(un_list):
                for nei_id in neigh:
                    p[i,ind] += Prob_mat[idx,nei_id]
            ind += 1
        p += 0.0001
        for i in range(len(un_list)):
            p[i] /= np.sum(p[i])
            exp_num[i] = np.sum(p[i]*length)
        entropy = -np.sum(p*np.log(p),axis=1)
        info = entropy/exp_num
        ind = np.argsort(info)[::-1]
        chosen_samples = [un_list[s] for s in ind[:batchsize]]

        # query the relation between the selected sample with the samples in neighborhood
        for cur_id in chosen_samples:
            l_list.append(cur_id)
            un_list.remove(cur_id)
            ind = np.argsort(p[i])[::-1]
            bingo = False
            for i in range(l):
                sample = list(certain_neighbor_dict[key_order[ind[i]]])[0]
                if state[cur_id,sample] == 0:
                    count += 1
                if query(targetIds,cur_id,sample) == 1:
                    state[cur_id,sample] = 1
                    certain_neighbor_dict[int(targetIds[sample])].add(cur_id)
                    state_resolution([cur_id,sample])
                    bingo = True
                    break
                else:
                    state[cur_id,sample] = -1
                    state_resolution([cur_id,sample])
            if not bingo:
                certain_neighbor_dict[int(targetIds[cur_id])] = {cur_id}
        
        if count > checkpoint:
            # perform semi-supervised clustering
            clusterer = PCKMeans(n_clusters=k,max_iter=50)
            clusterer.fit(features,ml=ml,cl=cl)
            kmeans_label = clusterer.labels_
            nmi,ari = test(targetIds,kmeans_label)
            nmi_record.append(nmi)
            ari_record.append(ari)
            if len(query_threshold_list):
                checkpoint = query_threshold_list.pop(0)
            else:
                break
    result = [checkpoint_list_,nmi_record,ari_record]
    print(result)
    end_time = time.time()
    print('total running time of NPU: {}seconds.'.format(end_time-start_time))

N = n_list[args.n]
checkpoint_list = [int(2000/6*i) for i in range(1,7)]
checkpoint_list_ = deepcopy(checkpoint_list)
NPU(checkpoint_list, batchsize=3)