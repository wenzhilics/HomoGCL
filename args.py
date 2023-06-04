import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index. -1 for cpu')
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of HomoGCL.')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay of HomoGCL.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
    parser.add_argument('--epoch1', type=int, default=500, help='Training epochs.')
    parser.add_argument('--epoch2', type=int, default=2000, help='Evaluation epochs.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--der', type=float, default=0.4, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.1, help='Drop feature ratio.')
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')
    parser.add_argument("--proj_dim", type=int, default=0, help='Project dim.')
    parser.add_argument("--tau", type=float, default=0.5, help='Temperature')
    parser.add_argument('--mean', action="store_true", help='Calculate mean for neighbor pos')
    parser.add_argument("--nclusters", type=int, default=5, help='Number of clusters in kmeans')
    parser.add_argument("--niter", type=int, default=20, help='Number of iteration for kmeans.')
    parser.add_argument("--sigma", type=float, default=1e-3, help='2sigma^2 in GMM')
    parser.add_argument("--alpha", type=int, default=1, help='Coefficient alpha')
    parser.add_argument("--clustering", action='store_true', default=False, help='Downstream clutering task.')
    parser.add_argument("--repetition_cluster", type=int, default=10, help='Repetition of clustering')
    args = parser.parse_args()

    return args
