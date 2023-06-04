import torch as th
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import faiss

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())

def homo_loss(x, edge_index, nclusters, niter, sigma):
    kmeans = faiss.Kmeans(x.shape[1], nclusters, niter=niter) 
    kmeans.train(x.cpu().detach().numpy())
    centroids = th.FloatTensor(kmeans.centroids).to(x.device)
    logits = []
    for c in centroids:
        logits.append((-th.square(x - c).sum(1)/sigma).view(-1, 1))
    logits = th.cat(logits, axis=1)
    probs = F.softmax(logits, dim=1)
    loss = F.mse_loss(probs[edge_index[0]], probs[edge_index[1]])
    return loss, probs

def evaluate_clustering(emb, nb_class, true_y, repetition_cluster):
    embeddings = F.normalize(emb, dim=-1, p=2).detach().cpu().numpy()

    estimator = KMeans(n_clusters = nb_class)

    NMI_list = []
    ARI_list = []

    for _ in range(repetition_cluster):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        nmi_score = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(true_y, y_pred)
        NMI_list.append(nmi_score)
        ARI_list.append(ari_score)

    return np.mean(NMI_list), np.std(NMI_list), np.mean(ARI_list), np.std(ARI_list)
