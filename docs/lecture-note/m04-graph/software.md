
# Software for Network Embedding

There are various software packages for network embeddings. But due to technical complexity, some of them do not faithfully implement the algorithms in the paper. We provide a list of software packages for network embeddings below.

- [fastnode2vec](https://github.com/louisabraham/fastnode2vec). This is a very fast implementation of node2vec. However, it uses a uniform probability distribution for the negative sampling, which is different from the original node2vec paper that uses a different distribution. This leads to some degeneracy of the embedding quality in community detection tasks.
- [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric). This is a very popular package for graph neural networks. It also uses a uniform probability distribution for the negative sampling, potentially having the same issue as `fastnode2vec`.
- [gnn-tools](https://github.com/skojaku/gnn-tools). This is a collection of my experiments on network embedding methods.
- [My collection](https://github.com/skojaku/graphvec). This is a lighter version of the `gnn-tools` collection.
