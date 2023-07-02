# HomoGCL: Rethinking Homophily in Graph Contrastive Learning

Implementation of KDD'23 paper HomoGCL: [Rethinking Homophily in Graph Contrastive Learning](https://arxiv.org/abs/2306.09614).

## Requirements

This repository has been tested with the following packages:
- Python == 3.8.12
- PyTorch == 1.11.0
- DGL == 0.8.2
- faiss == 1.7.2

## Important Hyperparameters

- `dataname`: Name of the dataset. Could be one of `[cora, citeseer, pubmed, photo, comp]`. The datasets will be downloaded automatically to `~/.dgl/` when run the code for the first time.
- `nclusters`:Number of clusters (centroids) in k-means. Default is 5.
- `alpha`: Weight coefficient between contrastive loss and homophily loss. Default is 1.
- `lr1`: Learning rate for HomoGCL.
- `lr2`: Learning rate for linear evaluator.
- `der`: Edge drop rate. Default is 0.4.
- `dfr`: Feature drop rate. Default is 0.1.
- `clustering`: Whether to do the downstream node clustering task. Default is False.

Please refer to [args.py](args.py) for the full hyper-parameters.

## How to Run

Pass the above parameters to `main.py`. For example:

```
python main.py --dataname cora --nclusters 10 --alpha 1 --epoch1 50 --mean --epoch2 1000
```

## Acknowledgements

The code is implemented based on [CCA-SSG](https://github.com/hengruizhang98/CCA-SSG).

## Citation

If you find this work is helpful to your research, please consider citing our paper:

```
@article{li2023homogcl,
  title={HomoGCL: Rethinking Homophily in Graph Contrastive Learning},
  author={Wen-Zhi Li and Chang-Dong Wang and Hui Xiong and Jian-Huang Lai},
  journal={arXiv preprint arXiv:2306.09614},
  year={2023}
}
```
