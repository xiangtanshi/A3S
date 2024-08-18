from Data_feature import *
from metric import *
from copy import deepcopy
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='The hyper-parameter setting for active probabilistic clustering.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=1, help='the data_path index')
parser.add_argument('--n', type=int, default=1, help='the N index')
parser.add_argument('--iso', type=int, default=1, help='the isotonic regressor index')
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

    # a slow approach 
    # l2_dists_square = np.zeros((N,N))
    # for i in range(N):
    #     delta = features[i] - features
    #     dist_i = np.sum(delta**2,axis=1)
    #     l2_dists_square[i] = dist_i
    # iso_reg_model = joblib.load(iso_paths[args.iso])
    # Prob_mat = iso_reg_model.predict(-l2_dists_square.flatten()).reshape(l2_dists_square.shape)
    # Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)

    # a faster approach without resorting to parallization, requires large memory yet
    iso_reg_model = joblib.load(iso_paths[args.iso])
    norm_squared = np.sum(features**2, axis=1, keepdims=True)
    inner_product = np.dot(features, features.T)
    inner_product = norm_squared - 2 * inner_product + norm_squared.T

    Prob_mat = iso_reg_model.predict(-inner_product.flatten()).reshape(inner_product.shape)
    Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)
    norm_squared = 0
    inner_product = 0
    

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

classes = len(np.unique(pclus_label))

# --------------------------------------------------------------------------------------
# define the necessary data structure and function
count = 0  # number of human queries
state = np.eye(N)  # query state matrix: {must-link:1, cannot-link:-1, not-queried: 0}
ML, CL = [], []    # must-link and cannot link pairs
leader_dict = dict()  # {cluster_id: sample_id}, sample_id is the sample that belongs to the dominant class in the cluster 
count_record = []
nmi_record = []
ari_record = []

def test(gt_label, pred_label):
    nmi=NMI(gt_label, pred_label)
    ari=ARI(gt_label, pred_label)
    purity=Purity(gt_label, pred_label)
    nmi,ari,purity = round(nmi,4), round(ari,4), round(purity,4)
    count_record.append(count)
    nmi_record.append(nmi)
    ari_record.append(ari)
    print(count,nmi,ari,purity)

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
                        if (min(p,q),max(p,q)) not in ML:
                            ML.append((min(p,q),max(p,q)))
        if len(sep_set) and len(link_set):
            for p in link_set:
                for q in sep_set:
                    state[p,q] = -1
                    state[q,p] = -1
                    if (min(p,q),max(p,q)) not in CL:
                        CL.append((min(p,q),max(p,q)))

def knn_update():
    # update the k-neighbors and prob_mat, log_prob_dict with the constraints
    log_probs = 9.21 # np.log(0.9999/0.0001)
    for item in ML:
        if item not in log_prob_dict:
            log_prob_dict[item] = log_probs
            Prob_mat[item[0],item[1]] = Prob_mat[item[1],item[0]] = 0.9999
            continue

        if log_prob_dict[item] == log_probs:
            continue
        else:
            p,q = item
            Prob_mat[p,q] = Prob_mat[q,p] = 0.9999
            log_prob_dict[item] = log_probs
    
    for item in CL:
        if item not in log_prob_dict:
            p,q = item
            Prob_mat[p,q] = Prob_mat[q,p] = 0.0001
            log_prob_dict[item] = -log_probs
            if p in knn[q]:
                ind = np.where(knn[q]==p)[0]
                knn[q,ind] = q
            if q in knn[p]:
                ind = np.where(knn[p]==q)[0]
                knn[p,ind] = p
            continue
        
        if log_prob_dict[item] == -log_probs:
            continue
        else:
            p,q = item
            Prob_mat[p,q] = Prob_mat[q,p] = 0.0001
            log_prob_dict[item] = -log_probs
            if p in knn[q]:
                ind = np.where(knn[q]==p)[0]
                knn[q,ind] = q
            if q in knn[p]:
                ind = np.where(knn[p]==q)[0]
                knn[p,ind] = p

def void_class():
    for key in range(classes,3*classes):
        if len(cluster_dict[key]) == 0:
            return key
        
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

def entropy(dists):
    H = 0
    for item in dists:
        H -= item[1] * item[0] / N * np.log(item[0] / N)
    return H


# --------------------------------------------------------------------------------------
# calculate the initial clustering result: recalculate pclus_label and cluster_dict

info = dict_info(name='pred')
K = 0
for item in info:
    if item[0] != 1:
        K += item[1]
print('adaptive cluster number:{}'.format(K))
if args.clus == 'kmeans':
    kmeans = KMeans(init="k-means++", n_clusters=int(K*2), n_init=4, random_state=0)
    kmeans = kmeans.fit(features)
    pclus_label = kmeans.labels_
elif args.clus == 'spec':
    Affinity = deepcopy(Prob_mat)
    # Affinity[Affinity<0.1] = 0
    pclus_label = spectral_clustering(Affinity, n_clusters=K, eigen_solver="arpack")
elif args.clus == 'aggo':
    aggo = AgglomerativeClustering(n_clusters=K, linkage="average", metric="euclidean")
    aggo.fit(features)
    pclus_label = aggo.labels_

cluster_dict = dict()
for i in range(2*classes):
    cluster_dict[i] = set()
for i in range(N):
    cluster_dict[pclus_label[i]].add(i)
# add placeholders for void class during active aggregation and splitting
for i in range(2*classes,3*classes):
    cluster_dict[i] = set()

# test the initial clustering result
test(targetIds,pclus_label)
info = dict_info(name='pred')
h = entropy(info)
print(info)
K = 0
o = 0
for item in info:
    if item[0] == 1:
        o = item[1]
    else:
        K += item[1]
print("Cluster num:{}, outliers num:{}, fission ratio:{}, entropy:{}".format(K,o,K/k,h))

# --------------------------------------------------------------------------------------
# define functions of BMAAS

# <functions related to calculate log-probabilistic-likelihood>

def get_log_likelihood(ind):
    # calculate the loglikelihood of sample ind in its current cluster
    pile = list(cluster_dict[pclus_label[ind]])
    log_llh = 0
    for i in pile:
        if i != ind:
            if (min(i,ind),max(i,ind)) in log_prob_dict:
                log_llh += log_prob_dict[(min(i,ind),max(i,ind))]
            else:
                log_prob_dict[(min(i,ind),max(i,ind))] = np.log(Prob_mat[i,ind]/(1-Prob_mat[i,ind]))
    return log_llh

def get_ll(ind, key):
    pile = list(cluster_dict[key])
    log_llh = 0
    for i in pile:
        log_llh += log_prob_dict[(min(i,ind),max(i,ind))]
    return log_llh/len(pile)

def ll_sort(pile):
    # sort a pile in descending order with their loglikelihoods
    pile_ll = [[i,get_log_likelihood(i)] for i in pile]
    pile_ll = sorted(pile_ll,key=lambda x: x[1], reverse=True)
    sorted_pile = [item[0] for item in pile_ll]
    return sorted_pile

def p_sort(i,pile):
    # sort a pile in descending order according to the pairwise probability between i and samples in pile
    pile_p = [[j,Prob_mat[i,j]] for j in pile]
    pile_p = sorted(pile_p,key=lambda x:x[1], reverse=True)
    sorted_pile = [item[0] for item in pile_p]
    return sorted_pile

# <functions related to cluster aggregation and splitting>

def adjust(i,j):
    # put i into j's cluster
    cluster_dict[pclus_label[j]].add(i)
    cluster_dict[pclus_label[i]].remove(i)
    pclus_label[i] = pclus_label[j]

def create_outlier_cluster(i):
    key = void_class()
    cluster_dict[key].add(i)
    cluster_dict[pclus_label[i]].remove(i)
    pclus_label[i] = key

def merge_class(key1,key2):
    # given that key2 and key2 are the same class, merge them to a single cluster
    pile = list(cluster_dict[key1])
    for i in pile:
        adjust(i,leader_dict[key2])

def lead_element_small(key,count):
    # judge if a small cluster (size<5) has leading class, if true, return the leader
    pile = list(cluster_dict[key])
    if len(pile) < 2:
        return -1,count
    pile = ll_sort(pile)
    if compactness(key, tau=args.tau):
        leader = pile[0]
        return leader,count
    else:
        # resort to human query
        i,j = pile[0],pile[1]
        if state[i,j] == 0:
            count += 1
        if query(targetIds,i,j):
            state[i,j] = 1
            leader = i
        else:
            state[i,j] = -1
            leader = -1
            create_outlier_cluster(j)
        state_resolution([i,j])
        knn_update()
        return leader,count  

def lead_element_big(key,count):
    # judge if a large cluster (size >=5) has leading class.
    # if true: return the leader; if not give the signal for purification
    pile = list(cluster_dict[key])
    pile = ll_sort(pile)
    if compactness(key,tau=args.tau):
        leader = pile[0]
        return leader,count
    else:
        if len(pile)<15:
            rate = 0.5
        else:
            rate = random.uniform(0.5,0.7)
        bingo = False
        mark = 0
        c = 0
        suspicious_set = []
        while not bingo:
            i = pile[c]
            c += 1
            j = pile[int(len(pile)*rate)]
            if state[i,j] == 0:
                count += 1
            if query(targetIds,i,j):
                state[i,j] = 1
                leader = i
                bingo = True
            else:
                state[i,j] = -1
                pile.remove(j)
                create_outlier_cluster(j)
                if c > 1: # at most query two times, and regard it as a impure cluster if no must-link is reached
                    print('cluster: {} is a mess cluster, could not find a leader!'.format(key))
                    mark = 1
                    leader = -1
                    bingo = True
                else:
                    suspicious_set.append(i)
            state_resolution([i,j])
            knn_update()

        return leader,count

def purification(key,count):
    # purify a mess cluster with greedy sample-based active query strategy
    pile = list(cluster_dict[key])
    certain_key_list = []
    max_cls_num = 3
    max_local_query = 20
    c = 0
    pile = ll_sort(pile)
    while len(pile):
        cur_id = pile.pop(0)
        if len(certain_key_list):
            # compare with newly assigned subclusters
            candidates = []
            # select a representative sample from each subclusters (max pairwise probability with cur_id)
            for i in certain_key_list:
                local = list(cluster_dict[i])
                candidates.append(local[0])
            candidates = p_sort(cur_id,candidates)
            # query cur_id with these candidates 
            for ids in candidates:
                if state[cur_id,ids] == 0:
                    count += 1
                    c += 1
                if query(targetIds,cur_id,ids):
                    state[cur_id,ids] = 1
                    adjust(cur_id,ids)
                    state_resolution([cur_id,ids])
                    knn_update()
                    break
                else:
                    state[cur_id,ids] = -1
                    state_resolution([cur_id,ids])
                    knn_update()
                    
        # create a void class for it when no local cluster fits cur_id
        if pclus_label[cur_id] == key:
            create_outlier_cluster(cur_id)
            leader_dict[pclus_label[cur_id]] = cur_id
            certain_key_list.append(pclus_label[cur_id])

        pile_cp = deepcopy(pile)
        pile_cp = p_sort(cur_id,pile_cp)
        tmp_count = 0
        for idx in pile_cp:
            if c > max_local_query:
                break
            if state[cur_id,idx] == 0:
                count += 1
                c += 1
            if query(targetIds,cur_id,idx):
                state[cur_id,idx] = 1
                adjust(idx,cur_id)
                pile.remove(idx)
                state_resolution([cur_id,idx])
                knn_update()
            else:
                state[cur_id,idx] = -1
                state_resolution([cur_id,idx])
                knn_update()
                if tmp_count < 3:
                    tmp_count += 1
                else:
                    break

        # completely separate the rest samples if more than max_cls_num subclusters are created
        if len(certain_key_list) > max_cls_num:
            for s in pile:
                create_outlier_cluster(s)
            if key in leader_dict:
                leader_dict.pop(key)
            return count
        
        if c > max_local_query and len(pile):
            # assign the rest samples to their cloest clusters in the certain_key_list
            for item in pile:
                belong_probs = [[certain_key, get_ll(item,certain_key)] for certain_key in certain_key_list]
                belong_probs = sorted(belong_probs,key=lambda x: x[1], reverse=True)
                adjust(item,leader_dict[belong_probs[0][0]])
            break

    if key in leader_dict:
        leader_dict.pop(key)

    return count

def outlier_merge(count):

    keys_list_1 = []
    for key in cluster_dict:
        if len(cluster_dict[key]) == 2:
            keys_list_1.append(key)
    for key in keys_list_1:
        s1,s2 = list(cluster_dict[key])
        if state[s1,s2] == 0:
            count += 1
        if query(targetIds,s1,s2):
            state[s1,s2] = 1
            state_resolution([s1,s2])
            leader_dict[key] = s1

            candidates = []
            for s in [s1,s2]:
                idx = 1
                nei = knn[s][idx]
                while nei == s and idx < args.neighbor:
                    idx += 1
                    nei = knn[s][idx]
                if pclus_label[nei] in leader_dict:
                    candidates.append(leader_dict[pclus_label[nei]])
            candidates = list(set(candidates))
            for candidate in candidates:
                if state[s1, candidate] == 0:
                    count += 1
                if query(targetIds, s1, candidate):
                    state[s1,candidate] = 1
                    state_resolution([s1,candidate])
                    knn_update()
                    adjust(s1,candidate)
                    adjust(s2,candidate)
                    break
                else:
                    state[s1,candidate] = -1
                    state_resolution([s1,candidate])
                    knn_update()
        else:
            state[s1,s2] = -1
            state_resolution([s1,s2])
            create_outlier_cluster(s1)

    keys_list_1 = []
    for key in cluster_dict:
        if len(cluster_dict[key]) == 1:
            keys_list_1.append(key)
    for key in keys_list_1:
        
        item = list(cluster_dict[key])[0]
        idx = 1
        nei = knn[item][idx]
        while nei == item and idx<args.neighbor:
            idx += 1
            nei = knn[item][idx]

        if pclus_label[nei] in leader_dict:
            # resort to human query for their cluster belongings
            leader = leader_dict[pclus_label[nei]]
            if state[item,leader] == 0:
                count += 1
            if query(targetIds,item,leader) == 1:
                adjust(item,leader)
                state[item,leader] = 1
            else:
                state[item,leader] = -1
            state_resolution([item,nei])
            knn_update()
    return count

# <functions related to query strategy>
def compactness(key, tau=0.8):
    # estimate the compactness of a cluster
    items = cluster_dict[key]
    num = len(items)
    k_neighbor = int(np.sqrt(num))
    compact = 0
    num_1 = 0
    for item in items:
        neigh_list = p_sort(item,items)
        neigh_list = neigh_list[int(num/2):int(num/2)+k_neighbor]
        num_1 += len(neigh_list)
        for item1 in neigh_list:
            compact += Prob_mat[item,item1]
    compact = compact / num_1
    if compact > tau:
        return 1
    else:
        return 0

def merging_probability(key1, key2):
    # estimates the probability that the two clusters actually are the same class
    if len(cluster_dict[key1]) <= len(cluster_dict[key2]):
        set1 = list(cluster_dict[key1])
        set2 = list(cluster_dict[key2])
    else:
        set1 = list(cluster_dict[key2])
        set2 = list(cluster_dict[key1])

    neighbor_num = 4

    P_list = []
    for i in set1:
        knn_probs = [Prob_mat[i,j] for j in set2]
        knn_probs.sort(reverse=True)
        P_list += knn_probs[0:neighbor_num]

    P_pos, P_neg = 1, 1
    for s in P_list:
        P_pos += s
        P_neg += 1-s
    P = P_pos/(P_pos + P_neg)

    return P

def delta_h(key1,key2):

    p = len(cluster_dict[key1])
    q = len(cluster_dict[key2])

    return 1 / N * ( (p+q) * np.log(p+q) - p*np.log(p) - q*np.log(q))

# --------------------------------------------------------------------------------------
# implement BMAAS
for t in range(args.T):
    key_list = []
    for key in cluster_dict.keys():
        if len(cluster_dict[key])>1:
            key_list.append(key)

    # recall cluster-pairs
    key2ent = []

    for i in range(len(key_list)-1):
        for j in range(i+1, len(key_list)):
            P = merging_probability(key_list[i],key_list[j])
            key2ent.append([key_list[i],key_list[j],P])

    key2ent = sorted(key2ent, key=lambda x: x[2], reverse=True)
    key2ent = key2ent[0:K]
    for items in key2ent:
        h = delta_h(items[0],items[1])
        items[2] *= h
    key2ent = sorted(key2ent, key=lambda x: x[2], reverse=True)
    print('The number of pairs selected:{}'.format(len(key2ent)))

    # perform aggregation and splitting
    while len(key2ent):

        item = key2ent.pop(0)
        key1, key2, _ = item

        # perform purity test and choose the leader for the dominant class
        if key1 in leader_dict:
            leader1 = leader_dict[key1]
            mark1 = 1
        else:
            if len(cluster_dict[key1]) > 4:
                leader1,count = lead_element_big(key1,count)
                if leader1 != -1:
                    # find the leader
                    mark1 = 1
                    leader_dict[key1] = leader1
                else:
                    # it is a mess cluster, purify it
                    count = purification(key1,count)
                    mark1 = 2
            else:
                # a simplified version of leader selection judgment
                leader1,count = lead_element_small(key1,count)
                if leader1 != -1:
                    #find the leader
                    mark1 = 1
                    leader_dict[key1] = leader1
                else:
                    mark1 = 2

        if key2 in leader_dict:
            leader2 = leader_dict[key2]
            mark2 = 1
        else:
            if len(cluster_dict[key2]) > 4:
                leader2,count = lead_element_big(key2,count)
                if leader2 != -1:
                    # find the leader
                    mark2 = 1
                    leader_dict[key2] = leader2
                else:
                    # it is a mess cluster, purify it
                    count = purification(key2,count)
                    mark2 = 2
            else:
                # a simplified version of leader selection judgment
                leader2,count = lead_element_small(key2,count)
                if leader2 != -1:
                    #find the leader
                    mark2 = 1
                    leader_dict[key2] = leader2
                else:
                    mark2 = 2
                
        if mark1 == 1 and mark2 == 1:
            if state[leader1,leader2] == 0:
                count += 1
            if query(targetIds,leader1,leader2):
                state[leader1,leader2] = 1
                state_resolution([leader1,leader2])
                knn_update()
                merge_class(key1,key2)
                leader_dict.pop(key1)
            else:
                state[leader1,leader2] = -1
                state_resolution([leader1,leader2])
                knn_update()

        # update the key2ent list
        for items in key2ent:
            if key1 in items and len(cluster_dict[key1]) < 2:
                key2ent.remove(items)
            elif key2 in items and len(cluster_dict[key2]) < 2:
                key2ent.remove(items)
        if count % 3 == 0: 
            test(targetIds, pclus_label)

# deal with the outlier samples
count = outlier_merge(count)
test(targetIds, pclus_label)
end_time = time.time()
print('total running time of A3S: {}seconds.'.format(end_time-start_time))

# test the final clustering result
info = dict_info(name='pred')
h = entropy(info)
print(info)
K = 0
o = 0
for item in info:
    if item[0] == 1:
        o = item[1]
    else:
        K += item[1]
print("Final result: Cluster num:{}, outliers num:{}, fission ratio:{}, entropy:{}".format(K,o,K/k,h))

info = dict_info(name='real')
h = entropy(info)
print(info)
K = 0
for item in info:
    K += item[1]
print("Ground truth result: entropy:{}".format(h))

length = len(count_record)
checkpoint_list = [int((length-1)/7*i) for i in range(8)]
count_record = [count_record[checkpoint] for checkpoint in checkpoint_list]
nmi_record = [nmi_record[checkpoint] for checkpoint in checkpoint_list]
ari_record = [ari_record[checkpoint] for checkpoint in checkpoint_list]
result = [count_record,nmi_record,ari_record]
print(result)
