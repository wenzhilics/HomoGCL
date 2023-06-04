import torch as th
import torch.nn as nn

from aug import random_aug
from dataset import load
from model import HomoGCL, LogReg
from utils import sim, homo_loss, evaluate_clustering
from args import parse_args

import warnings
warnings.filterwarnings('ignore')

args = parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

th.manual_seed(args.seed)

if __name__ == '__main__':

    print(args)
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    in_dim = feat.shape[1]
    N = graph.number_of_nodes()
    model = HomoGCL(in_dim, args.hid_dim, args.out_dim, args.n_layers, N, num_proj_hidden=args.proj_dim, tau=args.tau)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    # ground truth graph and feature
    graph_cuda = graph.to(args.device)
    graph_cuda = graph_cuda.remove_self_loop().add_self_loop()
    feat_cuda = feat.to(args.device)

    for epoch in range(1, args.epoch1+1):
        model.train()
        optimizer.zero_grad()

        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)
        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)
        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)       
        z1, z2, z = model(graph1, feat1, graph2, feat2, graph_cuda, feat_cuda)

        adj1 = th.zeros(N, N, dtype=th.int).to(args.device)
        adj1[graph1.remove_self_loop().edges()] = 1
        adj2 = th.zeros(N, N, dtype=th.int).to(args.device)
        adj2[graph2.remove_self_loop().edges()] = 1

        homoloss, homoprobs = homo_loss(z, graph_cuda.edges(), args.nclusters, args.niter, args.sigma)
        confmatrix = sim(homoprobs, homoprobs) # saliency matrix
        loss = model.loss(z1, adj1, z2, adj2, confmatrix, args.mean)
        loss = loss + args.alpha * homoloss

        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Node classification evaluation ===")
    embeds = model.get_embedding(graph_cuda, feat_cuda)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    logreg = LogReg(train_embs.shape[1], num_class)
    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch2+1):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)
            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)
            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
            print('Epoch: {}, train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(epoch, train_acc, val_acc, test_acc))

    if args.clustering:
        print("=== Node clustering evaluation ===")
        nmi, nmi_std, ari, ari_std = evaluate_clustering(embeds, num_class, labels, args.repetition_cluster)
        print('nmi:{:.4f}, nmi std:{:.4f}, ari:{:.4f}, ari std:{:.4f}'.format(nmi, nmi_std, ari, ari_std))
