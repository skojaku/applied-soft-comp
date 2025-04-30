---
marp: true
theme: default
paginate: true
---



# Graph Neural Networks 🧠
### From Images to Graphs

![bg right:40%](https://theaisummer.com/static/02e23adc75fe68e5dd249a94f3c1e8cc/c483d/graphsage.png)

---

# What is a graph?

- A set of nodes connected by edges
- Simple representation of a relationship between objects
- [Zoo of graphs!](https://skojaku.github.io/adv-net-sci/intro/zoo-of-networks.html)


---

![width:1800px](https://www.azquotes.com/picture-quotes/quote-give-a-small-boy-a-hammer-and-he-will-find-that-everything-he-encounters-needs-pounding-abraham-kaplan-54-61-55.jpg)

---

# Neural Networks ~ Our hammer!


---

# From Images to Graphs

- Image = 2D grid of pixels
- Through a convolution, a pixel value is influenced by its neighbors
- We can represent this neighborhood structure using a graph and define **convolutions on graphs**!

![bg right:55% width:100%](https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/conv_graph-thumbnail_webp-600x300.webp)

---

# Let's represent a graph mathematically

- Adjacency matrix $A$
- $A_{ij}=1$ if there is an edge between node $i$ and node $j$, otherwise $A_{ij}=0$

<center>
<img src="https://mathworld.wolfram.com/images/eps-svg/AdjacencyMatrix_1002.svg" width="700px">
</center>

---

# Convolution on images ~ Fourier Transform

# Convolution on graph ~ ?

---

# Convolution on images ~ Fourier Transform

# Convolution on graph ~ Eigenvalues of Laplacian

---

# Key idea

## 1D signal
- Suppose a 1D signal $x(t)$ as a function of time $t$.

- The frequency of the signal is essentially *the speed of variation*.

- High frequency signal ~ rapid variations

- Low frequency signal ~ slow variations

## 2D signal ~ Same as 1D signal

---

## 1D signal
- Suppose a 1D signal $x(t)$ as a function of time $t$.

- The frequency of the signal is essentially *the speed of variation*.

- High frequency signal ~ rapid variations

- Low frequency signal ~ slow variations

## 2D signal ~ Same as 1D signal

## What about graph?

---

- Graph is non-trivial since it does not have an inherent order of nodes! (like time dimension in 1D signal and spatial dimension in 2D signal)

- But we can still define the variation as the sum of differences between neighboring nodes.

![bg right:50% width:80%](https://i.sstatic.net/52ohy.png)

---

# Total variation

- Suppose we have a graph of $N$ nodes, each node has a feature $x_i$.

- **The total variation** measures the smoothness of the node features:

$$J = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N A_{ij}(x_i - x_j)^2 = {\bf x}^\top {\bf L} {\bf x}$$

where ${\bf L}$: Graph Laplacian, $x_i$: Node features, $A_{ij}$: Adjacency matrix.

<div class="admonition question">

**Q**: What $x$ makes the total variation smallest (most smooth) and largest (most varying), provided that the norm of $x$ is fixed? 🤔

</div>

---

<div class="admonition answer">

The eigendecomposition of the Laplacian:

$${\bf L}{\bf x} = \lambda {\bf x}$$

By multiplying both sides by ${\bf x}^\top$, we get

$${\bf x}^\top {\bf L} {\bf x} = \lambda$$


This tells us that:

1. The eigenvectors with small eigenvalues represent **low-frequency** signals.

2. The eigenvectors with large eigenvalues represent **high-frequency** signals.

</div>


---

# Decomposing the Total Variation

The total variation can be decomposed as follows (${\bf u}_i$ is the eigenvector of the Laplacian):

$$
\begin{aligned}
J &= {\bf x}^\top {\bf L} {\bf x} = {\bf x}^\top \left(\sum_{i=1}^N \lambda_i {\bf u}_i{\bf u}_i^\top \right)  {\bf x} = \sum_{i=1}^N \lambda_i ({\bf x}^\top {\bf u}_i)({\bf u}_i^\top {\bf x}) \\
  &= \sum_{i=1}^N \lambda_i \underbrace{||{\bf x}^\top {\bf u}_i||^2}_{\text{alignment between } {\bf x} \text{ and } {\bf u}_i}
\end{aligned}
$$

Key Insight:
- The total variation is now decomposed into the sum of different frequency components $\lambda_i \cdot ||{\bf x}^\top {\bf u}_i||^2$.
- $\lambda_i$ acts as *a filter (kernel)* that reinforces or passes the signal ${\bf x}^\top {\bf u}_i$.

---

<center>
<img src="https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png" width="80%">
</center>
