---
marp: true
theme: default
paginate: true
---


<style>
/* @theme custom */

/* Base slide styles */
section {

    width: 1280px;
    height: 720px;
    padding: 40px;
    font-size: 28px;
}

/* Marp Admonition Styles with Auto Labels */
.admonition {
    margin: 1em 0; /* Adjusted margin */
    padding: 1em 1em 1em 1em;
    border-left: 4px solid;
    border-radius: 3px;
    background: #f8f9fa;
    position: relative;
    line-height: 1.5;
    color: #333;
}

/* Add spacing for the title */
.admonition::before {
    display: block;
    margin-bottom: 0.8em;
    font-weight: 600;
    font-size: 1.1em;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Note */
.note {
    border-left-color: #0066FF;
}

.note::before {
    content: "ℹ️ Note";
    color: #0066FF;
}

/* Question */
.question {
    border-left-color: #00A550;
}

.question::before {
    content: "🤔 Question";
    color: #00A550;
}

/* Intuition */
.intuition {
    border-left-color: #9B59B6;
}

.intuition::before {
    content: "💭 Intuition";
    color: #9B59B6;
}


/* Answer */
.answer {
    border-left-color: #0066FF;
}

.answer::before {
    content: "💡 Answer";
    color: #0066FF;
}

/* Tip */
.tip {
    border-left-color: #00A550;
}

.tip::before {
    content: "💡 Tip";
    color: #00A550;
}

/* Important */
.important {
    border-left-color: #8B44FF;
}

.important::before {
    content: "📢 Important";
    color: #8B44FF;
}

/* Warning */
.warning {
    border-left-color: #CD7F32;
}

.warning::before {
    content: "⚠️ Warning";
    color: #CD7F32;
}

/* Caution */
.caution {
    border-left-color: #FF3333;
}

.caution::before {
    content: "🚫 Caution";
    color: #FF3333;
}

/* Two-Column Layout for Marp Slides */
/* Basic column container */
.columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
    height: calc(100% - 2rem); /* Account for margin-top */
}

/* Left column */
.column-left {
    grid-column: 1;
    color: inherit;
    min-width: 0; /* Prevent overflow */
}

/* Right column */
.column-right {
    grid-column: 2;
    color: inherit;
    min-width: 0; /* Prevent overflow */
}

/* Optional: Equal height columns with top alignment */
.columns-align-top {
    align-items: start;
}

/* Optional: Center-aligned columns */
.columns-align-center {
    align-items: center;
}

/* Optional: Different column width ratios */
.columns-40-60 {
    grid-template-columns: 40fr 60fr;
}

.columns-60-40 {
    grid-template-columns: 60fr 40fr;
}

.columns-30-70 {
    grid-template-columns: 30fr 70fr;
}

.columns-70-30 {
    grid-template-columns: 70fr 30fr;
}

/* Ensure images scale properly within columns */
.column-left img,
.column-right img {
    max-width: 100%;
    height: auto;
}

/* Optional: Add borders between columns */
.columns-divided {
    column-gap: 2rem;
    position: relative;
}

.columns-divided::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    border-left: 1px solid #ccc;
    transform: translateX(-50%);
}

/* Fix common Markdown elements inside columns */
.columns h1,
.columns h2,
.columns h3,
.columns h4,
.columns h5,
.columns h6 {
    margin-top: 0;
}

.columns ul,
.columns ol {
    padding-left: 1.5em;
}

/* Ensure code blocks don't overflow */
.columns pre {
    max-width: 100%;
    overflow-x: auto;
}
</style>


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

---

# Spectral Filtering 🎛️

## Low-pass Filter:
$$h_{\text{low}}(\lambda) = \frac{1}{1 + \alpha\lambda}$$

## High-pass Filter:
$$h_{\text{high}}(\lambda) = \frac{\alpha\lambda}{1 + \alpha\lambda}$$

Properties:
- Controls which frequencies pass
- Smooth or sharpen signals

![bg right:50% width:100%](https://skojaku.github.io/adv-net-sci/_images/4f8ad2e95bb844b5887d5d55053435d248c98042abf4727d6784dc191e07bf7f.png)

---

<div class="admonition question">

We learned that the total variation of a signal on a graph can be decomposed into the sum of different spectral components of the Laplacian.

$$
\begin{aligned}
J &= {\bf x}^\top {\bf L} {\bf x} = {\bf x}^\top \left(\sum_{i=1}^N \lambda_i {\bf u}_i{\bf u}_i^\top \right)  {\bf x} = \sum_{i=1}^N \lambda_i ({\bf x}^\top {\bf u}_i)({\bf u}_i^\top {\bf x}) \\
  &= \sum_{i=1}^N \lambda_i \underbrace{||{\bf x}^\top {\bf u}_i||^2}_{\text{alignment between } {\bf x} \text{ and } {\bf u}_i}
\end{aligned}
$$

But there is no *learnable* parameter yet. How would you design a *learnable* graph convolution 🤔?

</div>

---

<div class="admonition answer">

Replace the "eigenvalues" with "learnable parameters":

$$
{\bf L}_{\text{learn}} = \sum_{k=1}^K \theta_k {\mathbf u}_k {\mathbf u}_k^\top
$$

where ${\mathbf u}_k$ is the eigenvector of the Laplacian, and $\theta_k$ is a learnable parameter. Then apply the graph convolution:

$$
{\bf x}^{(\ell+1)} = h\left( L_{\text{learn}} {\bf x}^{(\ell)}\right)
$$
- $h$: Activation function
- $K$: Number of filters

This is called **Spectral Graph Convolution**! https://arxiv.org/abs/1312.6203


</div>

---

<div class="admonition question">

Now, let's consider the case of multi-dimensional features. Namely, each node now has a feature vector ${\bf x}_i$ instead of a scalar.

How would you define the learnable graph convolution in this case?

Scalar case:
$$
x_1 = 1.0, x_2 = 2.0, x_3 = 3.0
$$

Multi-dimensional case:
$$
{\bf x}_1 = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix}, {\bf x}_2 = \begin{bmatrix} 4.0 \\ 5.0 \\ 6.0 \end{bmatrix}, {\bf x}_3 = \begin{bmatrix} 7.0 \\ 8.0 \\ 9.0 \end{bmatrix}
$$


</div>

---

<div class="admonition answer">

For each pair $(i,j)$ of input and output features, we define a learnable filter:

$$
{\bf L}^{(i,j)}_{\text{learn}} = \sum_{k=1}^K \theta_{k, (i,j)} {\mathbf u}_k {\mathbf u}_k^\top
$$

Then, we apply the graph convolution:

$$
{\bf x}_i^{(j+1)} = h\left( \sum_{j=1}^{f_{\text{in}}} L_{\text{learn}}^{(i,j)} {\bf x}_j^{(j)}\right)
$$

</div>

![bg right:50% width:100%](https://d2l.ai/_images/conv-multi-in.svg)

---

<div class="admonition question">

Let's focus on the case of scalar features.

$$
{\bf L}_{\text{learn}} = \sum_{k=1}^K \theta_k {\mathbf u}_k {\mathbf u}_k^\top
$$

$$
{\bf x}^{(\ell+1)} = h\left( L_{\text{learn}} {\bf x}^{(\ell)}\right)
$$

What is the time and space complexity of spectral graph convolution for a graph with $N$ nodes and $E$ edges?

</div>

---

<div class="admonition answer">

The time complexity is ${\cal O}(N^3)$ as it involves the eigendecomposition of the Laplacian matrix.

The space complexity is ${\cal O}(N^2)$ as it involves the storage of the Laplacian matrix.



</div>

---
# Limitations of Spectral GNNs ⚠️

1. **Computational Cost**:
   - Eigendecomposition: $O(N^3)$
   - Prohibitive for large graphs

2. **Spatial Locality**:
   - Non-localized filters
   - Global node influence

---

# ChebNet: A Bridge Solution 🌉

**Key Idea**: Approximate filters using Chebyshev polynomials

$${\bf L}_{\text{learn}} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{{\bf L}})$$

where:
- $T_k$: Chebyshev polynomials, i.e., $T_0(x) = 1$, $T_1(x) = x$, $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$
- $\tilde{{\bf L}}$: Scaled Laplacian, i.e., $\tilde{{\bf L}} = 2\lambda_{\text{max}}^{-1}{\bf L} - {\bf I}$
- **Key property**:
  - A node can influence other nodes within $K$ hops away.
  - Faster computation

- https://arxiv.org/abs/1606.09375

---

# Interactive Demo

- https://marimo.io/p/@sadamori-kojaku/notebook-7qsxeq

---

# From Spectral to Spatial GNNs 🔄

**Evolution of Graph Convolutional Networks**:
1. Spectral GNNs (full eigen) ➡️ https://arxiv.org/abs/1312.6203
2. ChebNet (polynomial approx) ➡️ https://arxiv.org/abs/1606.09375
3. ▶️ ***GCN (first-order approximation)*** ➡️ https://arxiv.org/abs/1609.02907
4. Modern spatial GNNs

---

# Graph Convolutional Networks 🌐
### A Simple Yet Powerful Architecture
#### Kipf & Welling (2017)

![bg right:40%](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_7.44.34_PM.png)

---

<div class="admonition question">

What is the most simple form of ChebNet?

**ChebNet's Formulation**:
$${\bf L}_{\text{learn}} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{{\bf L}})$$

where:
- $T_k$: Chebyshev polynomials, i.e., $T_0(x) = 1$, $T_1(x) = x$, $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$
- $\tilde{{\bf L}}$: Scaled Laplacian, i.e., $\tilde{{\bf L}} = 2\lambda_{\text{max}}^{-1}{\bf L} - {\bf I}$

</div>



---

# From ChebNet to GCN 🔄

## ChebNet's Formulation:
$$h_{\theta'} * x \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{{\bf L}})x$$

## First-Order Approximation:
$$g_{\theta'} * x \approx \theta'_0x + \theta'_1(L - I_N)x = \theta'_0x - \theta'_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x$$

## Further Simplification $\theta = \theta_0 = -\theta_1$:
$$g_{\theta} * x \approx \theta(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$$

---

# Recipe of GCN  📊

- **Step 1**: Add self-loops to the adjacency matrix.
   $$A' = A + I_N$$

- **Step 2**: Compute the normalized adjacency matrix.

    $$ \tilde{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$$

    where $\tilde{D}_{ii} = \sum_j A'$ is the degree matrix of $A'$.

- **Step 3**: Perform the convolution.

    $$ Z = \tilde{A} X^{(\ell)}$$

- **Step 4**: Let different dimensions communicate with each other.

    $$ Z' = Z W^{(\ell)} $$

- **Step 5**: Apply non-linear activation.

    $$X^{(\ell+1)} = \sigma(Z')$$

---

# Renormalization Trick 🔧

- GCN is a powerful when it is deep (multiple convolutions).

- But, as the depth increases, the training becomes unstable due to **vanishing/exploding gradients**.

**Solution**: Add self-connections with renormalization:

$$\tilde{A} = A + I_N$$
$$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$


![bg right:40% width:100%](https://raw.githubusercontent.com/mbernste/mbernste.github.io/master/images/GCN_vs_CNN_overview.png)

---


# Extensions and Variants 🔄

1. **GraphSAGE**
   - Sampling-based approach
   - Inductive capability

2. **GAT**
   - Attention mechanisms
   - Weighted aggregation

3. **GIN**
   - Theoretically more powerful
   - WL test equivalent

---

# GraphSAGE = GCN + Sampling
**Key idea**:
- Sample a fixed number of nodes within the neighborhood for each node based on random walks.
- Allow for localized computation for each node.
- Efficient and scalable to large graphs

<img src="https://theaisummer.com/static/02e23adc75fe68e5dd249a94f3c1e8cc/c483d/graphsage.png" alt="GraphSAGE" width="75%" style="display: block; margin: 0 auto;">

---


**Another Key idea**: Aggregation function
- Treat the self node feature and the neighborhood features differently

$$
h_v^{(k+1)} = \text{CONCAT} \left(\underbrace{h_v^{(k)}}_{\text{self node feature}}, \underbrace{\text{AGGREGATE}}_{\text{Sum/Mean/Max/LTCM}}\left\{h_u^{(k)} \mid u \in \mathcal{N}(v)\right\}\right)
$$

<img src="https://theaisummer.com/static/02e23adc75fe68e5dd249a94f3c1e8cc/c483d/graphsage.png" alt="GraphSAGE" width="75%" style="display: block; margin: 0 auto;">

---

<div class="admonition question">

$$
h_v^{(k+1)} = \text{CONCAT} \left(\underbrace{h_v^{(k)}}_{\text{self node feature}}, \underbrace{\text{AGGREGATE}}_{\text{Sum/Mean/Max/LTCM}}\left\{h_u^{(k)} \mid u \in \mathcal{N}(v)\right\}\right)
$$

In GraphSAGE, AGGREGATE is a function that takes a list of node features and returns a single feature vector.
There are many choices of AGGREGATE functions.

- mean
- max/min
- sum

Aggregation entails the loss of information. Which one among the above is most likely to preserve the information of the neighborhood?

</div>



---

# Graph Isomorphism Network (GIN) 🔍

$$h_v^{(k+1)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot h_v^{(k)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k)}\right)$$

Key Features:
- Maximally powerful GNNs
- Theoretical connections to graph isomorphism
- Learnable or fixed $\epsilon$

---

# Graph Attention Networks (GAT) 👀

Attention Mechanism:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$e_{ij} = \text{LeakyReLU}(\vec{a}^\top [\underbrace{W h_i^{(k)}}_{\text{self node feature}}, \underbrace{W h_j^{(k)}}_{\text{neighbor node feature}}]) $$

Features:
- Learn importance of neighbors
- Multiple attention heads
- Dynamic edge weights

![bg right:50% width:100%](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_7.55.32_PM_vkdDcDx.png)


---


# Summary 📝

Key Takeaways:
- GNNs extend CNNs to irregular structures
- Multiple architectures available:
  - GCN: Simple and effective
  - GraphSAGE: Scalable and inductive
  - GAT: Attention-based
  - GIN: Theoretically powerful

Future Directions:
- Scalability
- Expressiveness
- Applications