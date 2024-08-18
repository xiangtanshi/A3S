## A3S: A General Active Clustering Method with Pairwise Constraints

This repository contains the code and the necessary data, monotonic regression model for our paper: [A3S: A General Active Clustering Method with Pairwise Constraints](https://web3.arxiv.org/pdf/2407.10196). We introduce A3S, a simple yet effective active clustering algorithm that significantly enhances clustering results by incorporating a small number of human-annotated pairwise labels (i.e., must-link or cannot-link constraints). A3S outperforms existing active clustering algorithms in a more practical scenarios where lots of samples from numerous classes are gathered for clustering (A3S is not recommended for small datasets with 2 or 3 classes). 


### Tips for Running A3S
As A3S incorporates several tricky techniques, we offer comprehensive guidance on its implementation. This detailed approach aims to facilitate your ability to apply A3S to your own datasets and conduct comparative analyses with other methods. Our step-by-step instructions should enable you to navigate the intricacies of A3S and fully leverage its capabilities in your research or applications. You can download the datasets used in the paper through this link: https://www.dropbox.com/scl/fo/y35yh2zeaepe14cawqf3q/AEE2Ad88y3dZQAWuvEGx6Aw?rlkey=g7sazn0qc45nc37awng5mqcy1&st=vpw1olm8&dl=0.

#### Environment
You need to create a python environment that contains the following package in order to run the files in the repository:
- sklearn
- joblib
- faiss (optional)
- base64
- tqdm
- active-semi-supervised-clustering (this is for the baseline)

#### Adaptive Clustering

We adopt [Probabilistic Clustering (PC)](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_MPC_Multi-View_Probabilistic_Clustering_CVPR_2022_paper.pdf) to generate the adaptive cluster number and the initial clustering result. As the realization of PC is not open by its original author, we provide our implementation in this repository, which is composed of two parts: pairwise posterior probability estimation and fast probabilistic clustering.

- pairwise posterior probability estimation: $P(e_{i,j}=1|d_{i,j})$ (in Data_feature.py):
  - For the input data $X=\{x_i\}_{i=1}^{N}$, calculate k-nearest-neighbors for the data and then construct sample pairs ($x_i$, $x_j$), where $x_j$ is the k-nearest-neighbor of $x_i$. Use kmeans clustering to generate pseudo labels for the data, then assign matching labels for these sample pairs according to the pseudo label: 1 if their pseudo labels are identical and otherwise 0.
  - Use isotonic regression to learn a better pairwise probability P(e_ij=1|d_ij) according to the constructed sample pair.
  - Optional: use path-propagation and co-neighbor-propagation to enhance the pairwise probability.
- Fast Probabilistic Clustering (in prob_cluster.py)

As far as we know, the quality of the pairwise probability is closely related to the performance of FPC, and you may email the author of PC to get more detailed tricks for this estimation, especially in the multi-view scenarios. And for your own datasets, you need to first train a corresponding monotonic regressor to convert pairwise distances to pairwise probability, and then run A3S.py on your own datasets.

#### Active Aggregation and Splitting
All the details of A3S are in A3S.py, and the major functions are as follows:
- state_resolution() and knn_update() that refresh the must-link, cannot-link and knn after each query
- entropy() , compactness() and merging_probability(), which corresponds to the query strategy
- lead_element_small() and lead_element_big() that uncovers the representative sample which is from the dominant class of a cluster with very high probability
- purification() which split clusters with low purity
  
the command for running A3S for different datasets:

```
MK20: python A3S.py --d 0 --n 0 --iso 0 --tau 0.5
MK100: python A3S.py --d 1 --n 1 --iso 0 --tau 0.8
Handwritten: python A3S.py --d 2 --n 2 --types 1 --tau 0.8 --T 4
Humbi-Face: python A3S.py --d 4 --n 3  --iso 2 --tau 0.5 --T 3
MS1M-10k: python A3S.py --d 5 --n 4 --iso 4 --tau 0 --clus aggo
MS1M-100k: python A3S.py --d 6 --n 5 --iso 4 --tau 0 --clus aggo
```

for the baseline methods, you can run these files: URASC.py NPU.py, FFQS.py, COBRA.py
