---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Graph Convolutional Networks
We have seen that spectral filters give us a principled way to think about "convolution" on irregular graph structures, and controlling the frequency components brings out different aspects of the data. We now go one step further: instead of designing filters by hand, we can learn them from data for specific tasks.


## Spectral Graph Convolutional Networks

A simplest form of learnable spectral filter is given by

$$
{\bf L}_{\text{learn}} = \sum_{k=1}^K \theta_k {\mathbf u}_k {\mathbf u}_k^\top,
$$

where ${\mathbf u}_k$ are the eigenvectors and $\theta_k$ are the learnable parameters. The variable $K$ is the number of eigenvectors used (i.e., the rank of the filter). The weight $\theta_k$ is learned to maximize the performance of the task at hand.

Building on this idea, {footcite}`bruna2014spectral` added a nonlinearity to the filter and proposed a spectral convolutional neural network (GCN) by

$$
{\bf x}^{(\ell+1)} = h\left( L_{\text{learn}} {\bf x}^{(\ell)}\right),
$$

where $h$ is an activation function, and ${\bf x}^{(\ell)}$ is the feature vector of the $\ell$-th convolution. They further extend this idea to convolve on multidimensional feature vectors, ${\bf X} \in \mathbb{R}^{N \times f_{\text{in}}}$ to produce new feature vectors of different dimensionality, ${\bf X}' \in \mathbb{R}^{N \times f_{\text{out}}}$.

$$
\begin{aligned}
{\bf X}^{(\ell+1)}_i &= h\left( \sum_j L_{\text{learn}}^{(i,j)} {\bf X}^{(\ell)}_j\right),\quad \text{where} \quad L^{(i,j)}_{\text{learn}} = \sum_{k=1}^K \theta_{k, (i,j)} {\mathbf u}_k {\mathbf u}_k^\top,
\end{aligned}
$$

Notice that the learnable filter $L_{\text{learn}}^{(i,j)}$ is defined for each pair of input $i$ and output $j$ dimensions.


```{note}
Many GCNs simple when it comes to implementation despite the complicated formula. And this is one of my ways to learn GNNs. Check out the [Appendix for the Python implementation](appendix.md).

```

## From Spectral to Spatial

Spectral GCNs are mathematically elegant but have two main limitations:
1. **Computational Limitation**: Computing the spectra of the Laplacian is expensive ${\cal O}(N^3)$ and prohibitive for large graphs
2. **Spatial Locality**: The learned filters are not spatially localized. A node can be influenced by all other nodes in the graph.

These two limitations motivate the development of spatial GCNs.

### ChebNet

ChebNet {footcite}`defferrard2016convolutional` is one of the earliest spatial GCNs that bridges the gap between spectral and spatial domains.
The key idea is to leverage Chebyshev polynomials to approximate ${\bf L}_{\text{learn}}$ by

$$
{\bf L}_{\text{learn}} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{{\bf L}}), \quad \text{where} \quad \tilde{{\bf L}} = \frac{2}{\lambda_{\text{max}}}{\bf L} - {\bf I},
$$

where $\tilde{{\bf L}}$ is the scaled and normalized Laplacian matrix in order to have eigenvalues in the range of $[-1,1]$. The Chebyshev polynomials $T_k(\tilde{{\bf L}})$ transforms the eigenvalues $\tilde{{\bf L}}$ to the following recursively:

$$
\begin{aligned}
T_0(\tilde{{\bf L}}) &= {\bf I} \\
T_1(\tilde{{\bf L}}) &= \tilde{{\bf L}} \\
T_k(\tilde{{\bf L}}) &= 2\tilde{{\bf L}} T_{k-1}(\tilde{{\bf L}}) - T_{k-2}(\tilde{{\bf L}})
\end{aligned}
$$

We then replace ${\bf L}_{\text{learn}}$ in the original spectral GCN with the Chebyshev polynomial approximation:

$$
{\bf x}^{(\ell+1)} = h\left( \sum_{k=0}^{K-1} \theta_k T_k(\tilde{{\bf L}}){\bf x}^{(\ell)}\right),
$$

where:
- $T_k(\tilde{{\bf L}})$ applies the k-th Chebyshev polynomial to the scaled Laplacian matrix
- $\theta_k$ are the learnable parameters
- K is the order of the polynomial (typically small, e.g., K=3)

### Graph Convolutional Networks by Kipf and Welling

While ChebNet offers a principled way to approximate spectral convolutions, Kipf and Welling (2017) {footcite}`kipf2017semi` proposed an even simpler and highly effective variant called **Graph Convolutional Networks (GCN)**.


#### First-order Approximation

The key departure is to use the first-order approximation of the Chebyshev polynomials.

$$
g_{\theta'} * x \approx \theta'_0x + \theta'_1(L - I_N)x = \theta'_0x - \theta'_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$

This is crude approximation but it leads to a much simpler form, leaving only two learnable parameters, instead of $K$ parameters in the original ChebNet.

Additionally, they further simplify the formula by using the same $\theta$ for both remaining parameters (i.e., $\theta_0 = \theta$ and $\theta_1 = -\theta$). The result is the following convolutional filter:

$$
g_{\theta} * x \approx \theta(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
$$

While this is a very simple filter, one can stack multiple layers of convolutions to perform high-order graph convolutions.

#### Deep GCNs can suffer from over-smoothing

GCN models can be deep, and when they are too deep, they start suffering from an ill-posed problem called *gradient vanishing/exploding*, where the gradients of the loss function becomes too small or too large to update the model parameters. It is a common problem in deep learning.

To facilitate the training of deep GCNs, the authors introduce a very simple trick called *renormalization*. The idea is to add self-connections to the graph:

$$
\tilde{A} = A + I_N, \quad \text{and} \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}
$$

And use $\tilde{A}$ and $\tilde{D}$ to form the convolutional filter.

Altogether, this leads to the following layer-wise propagation rule:

$$X^{(\ell+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X^{(\ell)}W^{(\ell)})$$

where:
- $X^{(\ell)}$ is the matrix of node features at layer $\ell$
- $W^{(\ell)}$ is the layer's trainable weight matrix
- $\sigma$ is a nonlinear activation function (e.g., ReLU)

These simplifications offer several advantages:
- **Efficiency**: Linear complexity in number of edges
- **Localization**: Each layer only aggregates information from immediate neighbors
- **Depth**: Fewer parameters allow building deeper models
- **Performance**: Despite (or perhaps due to) its simplicity, it often outperforms more complex models

```{admonition} Exercise
:class: note

Let's implement a simple GCN model for node classification.
[Coding Exercise](../../../notebooks/exercise-m09-graph-neural-net.ipynb)
```


```{footbibliography}
```