from prob_cluster import *
# import faiss

def faiss_search_approx_knn(query, target, k, n_gpu):
    # for market-1501 datasets with normalized features in a sphere space
    cpu_index = faiss.IndexFlatIP(target.shape[1])

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = False
    co.usePrecomputed = False
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=n_gpu)
    try:
        gpu_index.add(target)
    except:
        raise ValueError('cannot load feature to GPU')
    dists, nbrs = gpu_index.search(query, k=k)
    del gpu_index

    return dists, nbrs

def compute_posterior_prob_by_l2_distance(model_path=None, Features=None, K=2, n_neighbors=100):
    # learn a mapping function: -l2_distance^2 -> pairwise probability with the isotonic regression
    # Features: (n_samples, n_features), K: number of ground_truth classes
    features = Features
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    dists, nbrs = neigh.kneighbors(features)
    # dists, nbrs = faiss_search_approx_knn(features, features, n_neighbors, 1)  # for very large datasets
    dists = dists**2
    X = - dists.flatten()

    if model_path:
        #get psuedo label
        kmeans_clus = KMeans(n_clusters=K,init='k-means++',n_init='auto').fit(features)
        labels = kmeans_clus.labels_
        y = np.asarray(labels[nbrs.flatten()] == labels[nbrs[:, [0]*n_neighbors]].flatten(), dtype=int)
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(X, y)
        # save the mapping function
        joblib.dump(iso_reg, model_path, compress=9)
        return 1
    else:
        raise ValueError('no path to store the isotonic regression model!')

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

data_paths = ['data/datasets/market20.txt', 'data/datasets/market100.txt', 'data/datasets/hd_mv1.npy', 
               'data/datasets/handwritten.mat','data/datasets/test_mm_humbi240.mat', 'data/datasets/ms1m.npz']
    
iso_paths = ['data/models/market_iso_reg_20.pkl', 'data/models/market_iso_reg_100.pkl', 'data/models/humbi_face_iso_reg.pkl', 'data/models/handwritten_v1_iso_reg.pkl', 'data/models/ms1m_iso_reg.pkl']

n_list = [351, 1650, 2000, 5600, 10000, 100000]

def get_feature_similarity(data_index, n_index, types=0):

    data_path = data_paths[data_index]
    N = n_list[n_index]

    if data_index in [0,1]: # MK20
        features, labels = get_feature_from_file(data_path)
    elif data_index == 3:
        data = loadmat(data_path)
        features = data['X'][0,1]
        labels = data['Y'].reshape(N)
    elif data_index == 4:
        data = loadmat(data_path)
        features = data['X'][0,0][0:N]
        labels = data['Y'][0:N].reshape(N)
    elif data_index == 5:
        data = np.load(data_path)
        features = data['features'][0:N]
        labels = data['labels'][0:N]
        # add noise
        noise = np.random.normal(0,0.04,size=(N,512))
        features = features + noise
        features = features/1.35 # approximately normalize the noised features
    elif data_index == 2:
        Prob_mat = np.load(data_path)
        data = loadmat('data/datasets/handwritten.mat')
        features = data['X'][0,1]
        labels = data['Y'].reshape(N)
    # shuffle the data
    shuf_ind = [i for i in range(N)]
    random.shuffle(shuf_ind)
    features = features[shuf_ind]
    targetIds = labels[shuf_ind]
    if types == 0:
        return features, targetIds
    else:
        Prob_mat = Prob_mat[shuf_ind,:]
        Prob_mat = Prob_mat[:,shuf_ind]
        Prob_mat = np.clip(np.asarray(Prob_mat,dtype=float),0.0001,0.9999)
        return Prob_mat, targetIds