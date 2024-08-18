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
parser.add_argument('--d', type=int, default=3, help='the data_path index')
parser.add_argument('--n', type=int, default=2, help='the N index')
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

# load the data
features,targetIds = get_feature_similarity(args.d, args.n)
N = n_list[args.n]
k = len(np.unique(targetIds))
state = np.eye(N)
ml,cl=[],[]
nmi_record = []
ari_record = []

# calculate the Probability Matrix
if args.d == 3:
    Prob_mat, targetIds = get_feature_similarity(2, args.n, 1)
else:
    l2_dists_square = np.zeros((N,N))
    for i in range(N):
        delta = features[i] - features
        dist_i = np.sum(delta**2,axis=1)
        l2_dists_square[i] = dist_i
    iso_reg_model = joblib.load(iso_paths[args.iso])
    Prob_mat = iso_reg_model.predict(-l2_dists_square.flatten()).reshape(l2_dists_square.shape)
    Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)

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

def get_knn(W=None, n_neighbors=20):
    knn = np.zeros((N,n_neighbors))
    for i in range(N):
        p = W[i]
        ind = np.argsort(p)[::-1]
        knn[i] = ind[:n_neighbors]
    knn = knn.astype(int)
    return knn

def Entropy_knn(candidates, knn, labels):
    # For acceleration, approximating the entropy of each sample in the knn scope

    record = np.zeros((len(candidates),k))
    for i,idx in enumerate(candidates):
        for j in range(1,knn.shape[1]):
            record[i,labels[knn[idx,j]]] += Prob_mat[idx,knn[idx,j]]
    # normalize the probability
    record += 0.0001
    for i in range(len(candidates)):
        record[i] /= np.sum(record[i])
        
    entropy = -np.sum(record * np.log(record),axis=1)
    order = np.argsort(entropy)[::-1]
    ind = order[:int(N/np.log(N)/2)].tolist()
    top_index = [candidates[i] for i in ind]
    top_entropy = [entropy[i] for i in ind]
    return top_index, top_entropy

def get_neighbor_rep(i, nei_dict):
    # select one representative samples (most similar to sample i) from each certain neighborhoods
    rep_list = []
    for key in nei_dict:
        similarity = [[j, Prob_mat[i,j]] for j in list(nei_dict[key])]
        similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
        rep_list.append(similarity[0])
    rep_list = sorted(rep_list, key=lambda x:x[1], reverse=True)
    candidates = [item[0] for item in rep_list]
    return candidates

def Spectral_Clustering(W, K=10, normalized=0):
        
    # Compute the degree matrix and the unnormalized graph Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Compute the matrix with the first K eigenvectors as columns based on the normalized type of L
    if normalized == 1:   ## Random Walk normalized version
        # Compute the inverse of the diagonal matrix
        D_inv = np.diag(1/np.diag(D))
        # Compute the eigenpairs of L_{rw}
        Lambdas, V = np.linalg.eig(np.dot(D_inv, L))
        # Sort the eigenvalues by their L2 norms and record the indices
        # ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
        ind = np.argsort(Lambdas)
        V_K = np.real(V[:, ind[:K]])
    elif normalized == 2:   ## Graph cut normalized version
        # Compute the square root of the inverse of the diagonal matrix
        D_inv_sqrt = np.diag(1/np.sqrt(np.diag(D)))
        # Compute the eigenpairs of L_{sym}
        Lambdas, V = np.linalg.eig(np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt))
        # Sort the eigenvalues by their L2 norms and record the indices
        # ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
        ind = np.argsort(Lambdas)
        V_K = np.real(V[:, ind[:K]])
        if any(V_K.sum(axis=1) == 0):
            raise ValueError("Can't normalize the matrix with the first K eigenvectors as columns! Perhaps the number of clusters K or the number of neighbors in k-NN is too small.")
        # Normalize the row sums to have norm 1
        V_K = V_K/np.reshape(np.linalg.norm(V_K, axis=1), (V_K.shape[0], 1))
    else:   ## Unnormalized version
        # Compute the eigenpairs of L
        Lambdas, V = np.linalg.eig(L)
        # Sort the eigenvalues by their L2 norms and record the indices
        # ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
        ind = np.argsort(Lambdas)
        V_K = np.real(V[:, ind[:K]])
        
    # Conduct K-Means on the matrix with the first K eigenvectors as columns
    # kmeans = KMeans(n_clusters=K, init='k-means++', n_init='auto', random_state=0).fit(V_K)
    # return kmeans.labels_
    return V_K, Lambdas[ind], V[:,ind]

def eigen_vector_perturbation(eigenval_ord, eigenvec_ord, p, q, num=10):
    # given the eigen decomposition result of W, return the perturbation result when perturb W[p,q] slightly
    pert_list = []
    for idx in range(1,num+1):
        tmp_coeff = np.zeros((N,1))
        for i in range(0,N):
            if i != idx:
                tmp_coeff[i] = (eigenvec_ord[p,idx] * eigenvec_ord[p,i] + eigenvec_ord[q,idx] * eigenvec_ord[q,i] - 
                    eigenvec_ord[p,idx] * eigenvec_ord[q,i] - eigenvec_ord[q,idx] * eigenvec_ord[p,i]) / (eigenval_ord[idx] - eigenval_ord[i]) 
        delta_vec_idx = np.matmul(eigenvec_ord, tmp_coeff)
        pert_list.append(delta_vec_idx)      
    return pert_list

def URASC_ori(query_threshold=100,batchsize=5,pert_num=10):
    # perform uncertainty reduction active spectral clustering methods
    certain_neighbor_dict = dict()
    un_list = [i for i in range(N)]
    count = 0
    knn = get_knn(W=Prob_mat, n_neighbors=20)
    # get initial spectral clustering result
    V_K,eig_val,eig_vec = Spectral_Clustering(Prob_mat,K=k,normalized=0)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=0).fit(V_K)
    spec_label = kmeans.labels_
    test(targetIds,spec_label)

    # initialize the certain-set
    init_sample = np.random.randint(0,N)
    certain_neighbor_dict[targetIds[init_sample]] = {init_sample}
    l_list = [init_sample]
    un_list.remove(init_sample)
    
    while count < query_threshold and len(un_list)>batchsize:
        # get sample indexs with top n/log(n) step scale
        indexs, entropy = Entropy_knn(un_list,knn,spec_label)
        uncertainty = []
        # calculate the uncertainty reduction gradient for these samples

        # first select one representative sample from each certain set
        rel_samples = []
        for key in certain_neighbor_dict:
            rel_samples.append(int(np.random.choice(list(certain_neighbor_dict[key]),1)))
        grad_list = []
        # second, calculate the gradient 
        for idx in indexs:
            pert_vec_list = []
            for n_id in rel_samples:
                pert_vec_list.append(eigen_vector_perturbation(eig_val, eig_vec, idx, n_id, num=pert_num))
            grad_sum = 0
            for i in range(pert_num):
                grad_vec = np.zeros((N,1))
                for j in range(len(rel_samples)):
                    grad_vec += pert_vec_list[j][i]
                grad_sum += np.linalg.norm(grad_vec)
            grad_list.append(grad_sum)
        # choose the samples with the largest uncertainty
        for i in range(len(indexs)):
            uncertainty.append([indexs[i], entropy[i]*grad_list[i]])
            
        uncertainty = sorted(uncertainty,key=lambda x: x[1],reverse=True)
        for i in range(batchsize):
            l_list.append(uncertainty[i][0])
            un_list.remove(uncertainty[i][0])

        # perform annotation to decide on sample neighbor attribution
        chosen_samples = [item[0] for item in uncertainty[0:batchsize]]
        while len(chosen_samples):
            cur_id = chosen_samples.pop(0)
            cur_id_neigh = get_neighbor_rep(cur_id, certain_neighbor_dict)
            create_new_neigh = 1
            for nei in cur_id_neigh:
                count += 1
                if query(targetIds,cur_id,nei):
                    certain_neighbor_dict[targetIds[nei]].add(cur_id)
                    create_new_neigh = 0
                    break
            if create_new_neigh:
                certain_neighbor_dict[targetIds[cur_id]] = {cur_id}
        
        # update adjency matriix according to annotation result, and perform constrained spectral clustering
        for i in l_list:
            for j in l_list:
                if query(targetIds,i,j):
                    Prob_mat[i,j] = Prob_mat[j,i] = 1
                else:
                    Prob_mat[i,j] = Prob_mat[j,i] = 0

        V_K,eig_val,eig_vec = Spectral_Clustering(Prob_mat,K=k,normalized=0)
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=0).fit(V_K)
        spec_label = kmeans.labels_
        print(count,end=',')
        print(time.time(),end=',')
        test(targetIds,spec_label)

    return l_list

def URASC(query_threshold_list, batchsize=3):
    start_time = time.time()
    # perform uncertainty reduction active spectral clustering methods
    certain_neighbor_dict = dict()
    un_list = [i for i in range(N)]
    count = 0
    knn = get_knn(W=Prob_mat, n_neighbors=20)
    # get initial spectral clustering result
    spec_label = spectral_clustering(Prob_mat, n_clusters=k, eigen_solver="arpack")
    nmi,ari = test(targetIds,spec_label)
    nmi_record.append(nmi)
    ari_record.append(ari)

    # initialize the certain-set
    init_sample = np.random.randint(0,N)
    certain_neighbor_dict[targetIds[init_sample]] = {init_sample}
    l_list = [init_sample]
    un_list.remove(init_sample)
    
    checkpoint = query_threshold_list.pop(0)
    while count <= checkpoint and len(un_list)>batchsize:
        # get sample indexs with top n/log(n) step scale
        indexs, entropy = Entropy_knn(un_list,knn,spec_label)
        uncertainty = []
        # calculate the uncertainty reduction gradient for these samples

        # first select one representative sample from each certain set
        rel_samples = []
        for key in certain_neighbor_dict:
            rel_samples.append(int(np.random.choice(list(certain_neighbor_dict[key]),1)))
        
        for i in range(len(indexs)):
            uncertainty.append([indexs[i], entropy[i]])

        uncertainty = sorted(uncertainty,key=lambda x: x[1],reverse=True)
        for i in range(batchsize):
            l_list.append(uncertainty[i][0])
            un_list.remove(uncertainty[i][0])

        # perform annotation to decide on sample neighbor attribution
        chosen_samples = [item[0] for item in uncertainty[0:batchsize]]
        while len(chosen_samples):
            cur_id = chosen_samples.pop(0)
            cur_id_neigh = get_neighbor_rep(cur_id, certain_neighbor_dict)
            create_new_neigh = 1
            for nei in cur_id_neigh:
                if state[cur_id,nei] == 0:
                    count += 1
                
                if query(targetIds,cur_id,nei):
                    state[cur_id,nei] = 1
                    certain_neighbor_dict[targetIds[nei]].add(cur_id)
                    create_new_neigh = 0
                    state_resolution([cur_id,nei])
                    break
                else:
                    state[cur_id,nei] = -1
                    state_resolution([cur_id,nei])
            if create_new_neigh:
                certain_neighbor_dict[targetIds[cur_id]] = {cur_id}
        
        # update adjency matriix according to annotation result, and perform constrained spectral clustering
        for i in l_list:
            for j in l_list:
                if query(targetIds,i,j):
                    Prob_mat[i,j] = Prob_mat[j,i] = 1
                else:
                    Prob_mat[i,j] = Prob_mat[j,i] = 0

        if count > checkpoint:
            spec_label = spectral_clustering(Prob_mat, n_clusters=k, eigen_solver="arpack")
            nmi,ari = test(targetIds,spec_label)
            nmi_record.append(nmi)
            ari_record.append(ari)
            if len(query_threshold_list):
                checkpoint = query_threshold_list.pop(0)
            else:
                break
    result = [checkpoint_list_,nmi_record,ari_record]
    print(result)
    end_time = time.time()
    print('total running time of URASC: {}seconds.'.format(end_time-start_time))
    return len(l_list)/N

checkpoint_list = [1] + [int(2000/6*i) for i in range(1,7)]
checkpoint_list_ = deepcopy(checkpoint_list)
ratio = URASC(checkpoint_list, batchsize=3)
print(ratio)