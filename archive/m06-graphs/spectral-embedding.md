---
title: "Network Embedding Concepts"
jupyter: advnetsci
execute:
    enabled: true
---

## What is Network Embedding?


::: {#fig-karate-to-embedding}

![](https://transformerswsz.github.io/2022/09/09/DeepWalk%E8%A7%A3%E8%AF%BB/karate_to_embedding.jpg)

This figure is taken from [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652).
:::

Networks are high-dimensional discrete data that can be difficult to analyze with traditional machine learning methods that assume continuous and smooth data. Network embedding is a technique to represent networks in low-dimensional continuous spaces, enabling us to apply standard machine learning algorithms.

The goal of network embedding is to map each node in a network to a point in a low-dimensional space (typically $\mathbb{R}^d$ where $d \ll N$) while preserving important structural properties of the network.

## Exercises

- [✍️ Pen and paper exercises](pen-and-paper/exercise.pdf)

## Spectral Embedding

### Network Compression Approach

Let us approach spectral embedding from the perspective of network compression.
Suppose we have an adjacency matrix $\mathbf{A}$ of a network.
The adjacency matrix is high-dimensional data, i.e., a matrix of size $N \times N$ for a network of $N$ nodes.
We want to compress it into a lower-dimensional matrix $\mathbf{U}$ of size $N \times d$ for a user-defined small integer $d < N$.
A good $\mathbf{U}$ should preserve the network structure and thus can reconstruct the original data $\mathbf{A}$ as closely as possible.
This leads to the following optimization problem:

$$
\min_{\mathbf{U}} J(\mathbf{U}),\quad J(\mathbf{U}) = \| \mathbf{A} - \mathbf{U}\mathbf{U}^\top \|_F^2
$$

where:

1. $\mathbf{U}\mathbf{U}^\top$ is the outer product of $\mathbf{U}$ and represents the reconstructed network.
2. $\|\cdot\|_F$ is the Frobenius norm, which is the sum of the squares of the elements in the matrix.
3. $J(\mathbf{U})$ is the loss function that measures the difference between the original network $\mathbf{A}$ and the reconstructed network $\mathbf{U}\mathbf{U}^\top$.

By minimizing the Frobenius norm with respect to $\mathbf{U}$, we obtain the best low-dimensional embedding of the network.

### Spectral Decomposition Solution

Consider the spectral decomposition of $\mathbf{A}$:

$$
\mathbf{A} = \sum_{i=1}^N \lambda_i \mathbf{u}_i \mathbf{u}_i^\top
$$

where $\lambda_i$ are eigenvalues (weights) and $\mathbf{u}_i$ are eigenvectors. Each term $\lambda_i \mathbf{u}_i \mathbf{u}_i^\top$ is a rank-one matrix that captures a part of the network's structure. The larger the weight $\lambda_i$, the more important that term is in describing the network.

To compress the network, we can select the $d$ terms with the largest weights $\lambda_i$. By combining the corresponding $\mathbf{u}_i$ vectors into a matrix $\mathbf{U}$, we obtain a good low-dimensional embedding of the network.

![](../figs/spectral-decomposition.jpg)

For a formal proof, please refer to the [Appendix section](./04-appendix.md).

### Modularity Embedding

In a similar vein, we can use the modularity matrix to generate a low-dimensional embedding of the network.
Let us define the modularity matrix $\mathbf{Q}$ as follows:

$$
Q_{ij} = \frac{1}{2m}A_{ij} - \frac{k_i k_j}{4m^2}
$$

where $k_i$ is the degree of node $i$, and $m$ is the number of edges in the network.

We then compute the eigenvectors of $\mathbf{Q}$ and use them to embed the network into a low-dimensional space just as we did for the adjacency matrix. The modularity embedding can be used to bipartition the network into two communities using a simple algorithm: group nodes with the same sign of the second eigenvector [@newman2006modularity].

### Laplacian Eigenmap

Laplacian Eigenmap [@belkin2003laplacian] is another approach to compress a network into a low-dimensional space. The fundamental idea behind this method is to position connected nodes close to each other in the low-dimensional space. This approach leads to the following optimization problem:

$$
\min_{\mathbf{U}} J_{LE}(\mathbf{U}),\quad J_{LE}(\mathbf{U}) = \frac{1}{2}\sum_{i,j} A_{ij} \| u_i - u_j \|^2
$$

::: {.column-margin}
**Derivation steps:**

Starting from $J_{LE}(\mathbf{U})$, we expand:

$$
\begin{aligned}
&= \frac{1}{2}\sum_{i,j} A_{ij} \left( \| u_i \|^2 - 2 u_i^\top u_j + \| u_j \|^2 \right) \\
&= \sum_{i} k_i \| u_i \|^2 - \sum_{i,j} A_{ij} u_i^\top u_j\\
&= \sum_{i,j} L_{ij} u_i^\top u_j
\end{aligned}
$$

where $L_{ij} = k_i$ if $i=j$ and $L_{ij} = -A_{ij}$ otherwise.

In matrix form: $J_{LE}(\mathbf{U}) = \text{Tr}(\mathbf{U}^\top \mathbf{L} \mathbf{U})$

See [Appendix](./04-appendix.md) for full details.
:::

The solution minimizes distances between connected nodes. Through algebraic manipulation (see margin), we can rewrite the objective as $J_{LE}(\mathbf{U}) = \text{Tr}(\mathbf{U}^\top \mathbf{L} \mathbf{U})$, where $\mathbf{L}$ is the graph Laplacian matrix:

$$
L_{ij} = \begin{cases}
k_i & \text{if } i = j \\
-A_{ij} & \text{if } i \neq j
\end{cases}
$$

By taking the derivative and setting it to zero:

$$
\frac{\partial J_{LE}}{\partial \mathbf{U}} = 0 \implies \mathbf{L} \mathbf{U} = \lambda \mathbf{U}
$$

**Solution**: The $d$ eigenvectors associated with the $d$ smallest eigenvalues of the Laplacian matrix $\mathbf{L}$.

::: {.column-margin}
The smallest eigenvalue is always zero (with eigenvector of all ones). In practice, compute $d+1$ smallest eigenvectors and discard the trivial one.
:::

## Neural Embedding

### Introduction to word2vec

Neural embedding methods leverage neural network architectures to learn node representations. Before discussing graph-specific methods, we first introduce *word2vec*, which forms the foundation for many neural graph embedding techniques.

word2vec is a neural network model that learns word embeddings in a continuous vector space. It was introduced by Tomas Mikolov and his colleagues at Google in 2013 [@mikolov2013distributed].

### How word2vec Works

word2vec learns word meanings from context, following the linguistic principle: "You shall know a word by the company it keeps" [@church1988word].

::: {.column-margin}
This phrase means a person is similar to those they spend time with. It comes from Aesop's fable *The Ass and his Purchaser* (500s B.C.): a man brings an ass to his farm on trial. The ass immediately seeks out the laziest, greediest ass in the herd. The man returns the ass, knowing it will be lazy and greedy based on the company it chose.

<iframe width="100%" height="200" src="https://www.youtube.com/embed/gQddtTdmG_8?si=x8DUQnll2Rnj8qkn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
:::

**The Core Idea**: Given a target word, predict its surrounding context words within a fixed window. For example, in:

> The quick brown fox jumps over a lazy dog

The context of *fox* (with window size 2) includes: *quick*, *brown*, *jumps*, *over*.

::: {.column-margin}
The window size determines how far we look around the target word. A window size of 2 means we consider 2 words before and 2 words after.
:::

**Why This Works**: Words appearing in similar contexts get similar embeddings. Consider:

- "The quick brown **fox** jumps over the fence"
- "The quick brown **dog** runs over the fence"
- "The **student** studies in the library"

Both *fox* and *dog* appear with words like "quick," "brown," and "jumps/runs," so they'll have similar embeddings. But *student* appears in completely different contexts (like "studies," "library"), so its embedding will be far from *fox* and *dog*. This is how word2vec captures semantic similarity without explicit supervision.

**Two Training Approaches**:

- **Skip-gram**: Given a target word → predict context words (we'll use this)
- **CBOW**: Given context words → predict target word

::: {.column-margin}
Skip-gram works better for small datasets and rare words. CBOW is faster to train.
:::

**Network Architecture**: word2vec uses a simple 3-layer neural network:

![](../figs/word2vec.png){width="70%" fig-align="center"}

- **Input layer** ($N$ neurons): One-hot encoding of the target word
- **Hidden layer** ($d$ neurons, $d \ll N$): The learned word embedding
- **Output layer** ($N$ neurons): Probability distribution over context words (via softmax)

The hidden layer activations become the word embeddings—dense, low-dimensional vectors that capture semantic relationships.

::: {.column-margin}
For a visual walkthrough of word2vec, see [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) by Jay Alammar.
:::

### Key Technical Components

#### The Computational Challenge

To predict context word $w_c$ given target word $w_t$, we compute:

$$
P(w_c | w_t) = \frac{\exp(\mathbf{v}_{w_c} \cdot \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{v}_w \cdot \mathbf{v}_{w_t})}
$$

::: {.column-margin}
This is the softmax function—it converts dot products into probabilities that sum to 1.
:::

**The problem**: The denominator sums over all vocabulary words (e.g., 100,000+ terms), making training prohibitively slow.

#### Two Solutions

**Hierarchical Softmax**: Organizes vocabulary as a binary tree with words at leaves. Computing probability becomes traversing root-to-leaf paths, reducing complexity from $O(|V|)$ to $O(\log |V|)$.

::: {#fig-hierarchical-softmax}
![](https://building-babylon.net/wp-content/uploads/2017/07/hs4-1536x834.png)
This figure is taken from [Hierarchical Softmax](https://building-babylon.net/2017/08/01/hierarchical-softmax/) by Building Babylon.
:::

**Negative Sampling**: Instead of normalizing over all words, sample a few "negative" (non-context) words. Contrast the target's true context with these random words. This approximates the full softmax much more efficiently.

::: {.column-margin}
Typically, 5-20 negative samples are enough. The model learns to distinguish true context words from random words.
:::

### What's Special About word2vec?

With word2vec, words are represented as dense vectors, enabling us to explore their relationships using simple linear algebra. This is in contrast to traditional natural language processing (NLP) methods, such as bag-of-words and topic modeling, which represent words as discrete units or high-dimensional vectors.

::: {#fig-word2vec-embedding}
![](https://miro.medium.com/v2/resize:fit:678/1*5F4TXdFYwqi-BWTToQPIfg.jpeg)
This figure is taken from [Word2Vec](https://medium.com/@laura.wj.w/word2vec-207131259292) by Laura Wang.
:::

word2vec embeddings can capture semantic relationships, such as analogies (e.g., *man* is to *woman* as *king* is to *queen*) and can visualize relationships between concepts (e.g., countries and their capitals form parallel vectors in the embedding space).

## Graph Embedding with word2vec

How can we apply word2vec to graph data? There is a critical challenge: word2vec takes sequences of words as input, while graph data are discrete and unordered. A solution to fill this gap is *random walk*, which transforms graph data into a sequence of nodes. Once we have a sequence of nodes, we can treat it as a sequence of words and apply word2vec.

::: {.column-margin}
Random walks create "sentences" from graphs: each walk is a sequence of nodes, just like a sentence is a sequence of words.
:::

### DeepWalk

![](https://dt5vp8kor0orz.cloudfront.net/7c56c256b9fbf06693da47737ac57fae803a5a4f/1-Figure1-1.png)

DeepWalk is one of the pioneering works to apply word2vec to graph data [@perozzi2014deepwalk]. It views the nodes as words and the random walks on the graph as sentences, and applies word2vec to learn the node embeddings.

More specifically, the method contains the following steps:

1. Sample multiple random walks from the graph.
2. Treat the random walks as sentences and feed them to word2vec to learn the node embeddings.

DeepWalk typically uses the skip-gram model with hierarchical softmax for efficient training.

### node2vec

node2vec [@grover2016node2vec] extends DeepWalk with **biased random walks** controlled by two parameters:

$$
P(v_{t+1} = x | v_t = v, v_{t-1} = t) \propto
\begin{cases}
\frac{1}{p} & \text{if } d(t,x) = 0 \text{ (return to previous)} \\
1 & \text{if } d(t,x) = 1 \text{ (close neighbor)} \\
\frac{1}{q} & \text{if } d(t,x) = 2 \text{ (explore further)} \\
\end{cases}
$$

where $d(t,x)$ is the shortest path distance from the previous node $t$ to candidate node $x$.

**Controlling Exploration**:

- Low $p$ → return bias (local revisiting)
- Low $q$ → outward bias (exploration)
- High $q$ → inward bias (stay local)

**Two Exploration Strategies**:

![](https://www.researchgate.net/publication/354654762/figure/fig3/AS:1069013035655173@1631883977008/A-biased-random-walk-procedure-of-node2vec-B-BFS-and-DFS-search-strategies-from-node-u.png)

- **BFS-like** (low $q$): Explore immediate neighborhoods → captures **community structure**
- **DFS-like** (high $q$): Explore deep paths → captures **structural roles**

![](https://miro.medium.com/v2/resize:fit:1138/format:webp/1*nCyF5jFSU5uJVdAPdf-0HA.png)

::: {.column-margin}
**Technical Note**: node2vec uses **negative sampling** instead of hierarchical softmax, which affects embedding characteristics [@kojaku2021neurips; @dyer2014notes].
:::

### LINE

LINE [@tang2015line] is another pioneering work to learn node embeddings by directly optimizing the graph structure. It is equivalent to node2vec with $p=1$, $q=1$, and window size 1.

::: {.column-margin}
Neural methods are less transparent, but recent work establishes equivalences to spectral methods under specific conditions [@qiu2018network; @kojaku2024network]. Surprisingly, DeepWalk, node2vec, and LINE are also provably optimal for the stochastic block model [@kojaku2024network].
:::

---
title: "Network Embedding Coding"
jupyter: advnetsci
execute:
    enabled: true
---

In this section, we implement the embedding methods discussed in the [concepts section](./01-concepts.md).

## Data Preparation

We will use the karate club network throughout this notebook.

```{python}
import numpy as np
import igraph
import matplotlib.pyplot as plt
import seaborn as sns

# Load the karate club network
g = igraph.Graph.Famous("Zachary")
A = g.get_adjacency_sparse()

# Get community labels (Mr. Hi = 0, Officer = 1)
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
g.vs["label"] = labels

# Visualize the network
palette = sns.color_palette().as_hex()
igraph.plot(g, vertex_color=[palette[label] for label in labels], bbox=(300, 300))
```

## Spectral Embedding

### Example: Spectral Embedding with Adjacency Matrix

Let us demonstrate spectral embedding with the karate club network.

```{python}
# Convert to dense array for eigendecomposition
A_dense = A.toarray()
```

```{python}
# Compute the spectral decomposition
eigvals, eigvecs = np.linalg.eig(A_dense)

# Find the top d eigenvectors
d = 2
sorted_indices = np.argsort(eigvals)[::-1][:d]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

# Plot the results
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x = eigvecs[:, 0], y = eigvecs[:, 1], hue=labels, ax=ax)
ax.set_title('Spectral Embedding')
ax.set_xlabel('Eigenvector 1')
ax.set_ylabel('Eigenvector 2')
plt.show()
```

Interestingly, the first eigenvector corresponds to the eigencentrality of the network, representing the centrality of the nodes.
The second eigenvector captures the community structure of the network, clearly separating the two communities in the network.

### Example: Modularity Embedding

We can use the modularity matrix to generate a low-dimensional embedding of the network.

```{python}
deg = np.sum(A_dense, axis=1)
m = np.sum(deg) / 2
Q = A_dense - np.outer(deg, deg) / (2 * m)
Q/= 2*m

eigvals, eigvecs = np.linalg.eig(Q)

# Sort the eigenvalues and eigenvectors
sorted_indices = np.argsort(-eigvals)[:d]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x = eigvecs[:, 0], y = eigvecs[:, 1], hue=labels, ax=ax)
ax.set_title('Modularity Embedding')
ax.set_xlabel('Eigenvector 1')
ax.set_ylabel('Eigenvector 2')
plt.show()
```

### Example: Laplacian Eigenmap

Let us first compute the Laplacian matrix and its eigenvectors.
```{python}
D = np.diag(np.sum(A_dense, axis=1))
L = D - A_dense

eigvals, eigvecs = np.linalg.eig(L)

# Sort the eigenvalues and eigenvectors
sorted_indices = np.argsort(eigvals)[1:d+1]  # Exclude the first eigenvector
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

# Plot the results
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x = eigvecs[:, 0], y = eigvecs[:, 1], hue=labels, ax=ax)
ax.set_title('Laplacian Eigenmap')
ax.set_xlabel('Eigenvector 2')
ax.set_ylabel('Eigenvector 3')
plt.show()
```

## Neural Embedding with word2vec

### Example: word2vec with Text

To showcase the effectiveness of word2vec, let's walk through an example using the `gensim` library.

```{python}
import gensim
import gensim.downloader
from gensim.models import Word2Vec

# Load pre-trained word2vec model from Google News
model = gensim.downloader.load('word2vec-google-news-300')
```

Our first example is to find the words most similar to *king*.

```{python}
# Example usage
word = "king"
similar_words = model.most_similar(word)
print(f"Words most similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")
```

A cool (yet controversial) application of word embeddings is analogy solving. Let us consider the following puzzle:

> *man* is to *woman* as *king* is to ___ ?

We can use word embeddings to solve this puzzle.

```{python}

# We solve the puzzle by
#
#  vec(king) - vec(man) + vec(woman)
#
# To solve this, we use the model.most_similar function, with positive words being "king" and "woman" (additive), and negative words being "man" (subtractive).
#
model.most_similar(positive=['woman', "king"], negative=['man'], topn=5)
```

The last example is to visualize the word embeddings.

```{python}
#| echo: false

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

countries = ['Germany', 'France', 'Italy', 'Spain', 'Portugal', 'Greece']
capital_words = ['Berlin', 'Paris', 'Rome', 'Madrid', 'Lisbon', 'Athens']

# Get the word embeddings for the countries and capitals
country_embeddings = np.array([model[country] for country in countries])
capital_embeddings = np.array([model[capital] for capital in capital_words])

# Compute the PCA
pca = PCA(n_components=2)
embeddings = np.vstack([country_embeddings, capital_embeddings])
embeddings_pca = pca.fit_transform(embeddings)

# Create a DataFrame for seaborn
df = pd.DataFrame(embeddings_pca, columns=['PC1', 'PC2'])
df['Label'] = countries + capital_words
df['Type'] = ['Country'] * len(countries) + ['Capital'] * len(capital_words)

# Plot the data
plt.figure(figsize=(12, 10))

# Create a scatter plot with seaborn
scatter_plot = sns.scatterplot(data=df, x='PC1', y='PC2', hue='Type', style='Type', s=200, palette='deep', markers=['o', 's'])

# Annotate the points
for i in range(len(df)):
    plt.text(df['PC1'][i], df['PC2'][i] + 0.08, df['Label'][i], fontsize=12, ha='center', va='bottom',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Draw arrows between countries and capitals
for i in range(len(countries)):
    plt.arrow(df['PC1'][i], df['PC2'][i], df['PC1'][i + len(countries)] - df['PC1'][i], df['PC2'][i + len(countries)] - df['PC2'][i],
              color='gray', alpha=0.6, linewidth=1.5, head_width=0.02, head_length=0.03)

plt.legend(title='Type', title_fontsize='13', fontsize='11')
plt.title('PCA of Country and Capital Word Embeddings', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
ax = plt.gca()
ax.set_axis_off()
```

We can see that word2vec places the words representing countries close to each other and so do the words representing their capitals. The country-capital relationship is also roughly preserved, e.g., *Germany*-*Berlin* vector is roughly parallel to *France*-*Paris* vector.

## Graph Embedding with word2vec

### DeepWalk: Learning Network Embeddings via Random Walks

DeepWalk treats random walks on a graph as "sentences" and applies word2vec to learn node embeddings. The key insight is that nodes appearing in similar contexts (neighborhoods) should have similar embeddings.

::: {.column-margin}
DeepWalk was introduced by Perozzi et al. (2014) and was one of the first methods to successfully apply natural language processing techniques to graph embedding.
:::

#### Step 1: Generate Random Walks

The first step is to generate training data for word2vec. We do this by sampling random walks from the network. Each random walk is like a "sentence" where nodes are "words".

Let's implement a function to sample random walks. A random walk starts at a node and repeatedly moves to a random neighbor until it reaches the desired length.

```{python}
def random_walk(net, start_node, walk_length):
    """
    Generate a random walk starting from start_node.

    Parameters:
    -----------
    net : sparse matrix
        Adjacency matrix of the network
    start_node : int
        Starting node for the walk
    walk_length : int
        Length of the walk

    Returns:
    --------
    walk : list
        List of node indices representing the random walk
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(net[cur].indices)

        if len(cur_nbrs) > 0:
            # Randomly choose one of the neighbors
            walk.append(np.random.choice(cur_nbrs))
        else:
            # Dead end - terminate the walk
            break

    return walk
```

::: {.column-margin}
In practice, we generate multiple walks per node to ensure each node appears in various contexts, which helps the model learn better representations.
:::

Now we generate multiple random walks starting from each node. We'll use 10 walks per node, each of length 50.

```{python}
n_nodes = g.vcount()
n_walkers_per_node = 10
walk_length = 50

walks = []
for i in range(n_nodes):
    for _ in range(n_walkers_per_node):
        walks.append(random_walk(A, i, walk_length))

print(f"Generated {len(walks)} random walks")
print(f"Example walk: {walks[0][:10]}...")  # Show first 10 nodes of first walk
```

#### Step 2: Train the Word2Vec Model

Now we feed the random walks to the word2vec model. The model will learn to predict which nodes appear together in the same walk, similar to how it learns which words appear together in sentences.

```{python}
from gensim.models import Word2Vec

model = Word2Vec(
    walks,
    vector_size=32,   # Dimension of the embedding vectors
    window=3,         # Maximum distance between current and predicted node
    min_count=1,      # Minimum frequency for a node to be included
    sg=1,             # Use skip-gram model (vs CBOW)
    hs=1,              # Use hierarchical softmax for training,
    workers = 1,
)
```

::: {.column-margin}
**Key Parameters:**

- `vector_size`: Higher dimensions capture more information but require more data and computation.
- `window`: Larger windows capture broader context but may dilute local structure.
- `sg=1`: Skip-gram predicts context from target. It works better for small datasets.
- `hs=1`: Hierarchical softmax is faster than negative sampling for small vocabularies.
:::

The `window` parameter is crucial. For a random walk `[0, 1, 2, 3, 4, 5, 6, 7]`, when `window=3`, the context of node 2 includes nodes `[0, 1, 3, 4, 5]` - all nodes within distance 3.

#### Step 3: Extract Node Embeddings

After training, we can extract the learned embeddings for each node from the model.

```{python}
# Extract embeddings for all nodes
embedding = np.array([model.wv[i] for i in range(n_nodes)])

print(f"Embedding matrix shape: {embedding.shape}")
print(f"First node embedding (first 5 dimensions): {embedding[0][:5]}")
```

#### Step 4: Visualize Embeddings

Let's visualize the learned embeddings in 2D using UMAP (Uniform Manifold Approximation and Projection). UMAP reduces the 32-dimensional embeddings to 2D while preserving the local structure.

::: {.column-margin}
UMAP is a dimensionality reduction technique that preserves both local and global structure better than t-SNE, making it ideal for visualizing high-dimensional embeddings.
:::

```{python}
#| echo: false
import umap
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool

# Reduce embeddings to 2D
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, metric="cosine")
xy = reducer.fit_transform(embedding)

output_notebook()

# Calculate node degrees for visualization
degrees = A.sum(axis=1).A1

# Create interactive plot
source = ColumnDataSource(data=dict(
    x=xy[:, 0],
    y=xy[:, 1],
    size=np.sqrt(degrees / np.max(degrees)) * 30,
    community=[palette[label] for label in g.vs["label"]]
))

p = figure(title="DeepWalk Node Embeddings (UMAP projection)", x_axis_label="UMAP 1", y_axis_label="UMAP 2")
p.scatter('x', 'y', size='size', source=source, line_color="black", color="community")

show(p)
```

Notice how nodes from the same community (shown in the same color) tend to cluster together in the embedding space. This demonstrates that DeepWalk successfully captures the community structure.

#### Step 5: Clustering with K-means

One practical application of node embeddings is clustering. While we have dedicated community detection methods like modularity maximization, embeddings allow us to use general machine learning algorithms like K-means.

::: {.column-margin}
The advantage of using embeddings is that we can leverage the rich ecosystem of machine learning tools designed for vector data.
:::

Let's implement K-means clustering with automatic selection of the number of clusters using the silhouette score.

```{python}
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_clusters(embedding, n_clusters_range=(2, 10)):
    """
    Find the optimal number of clusters using silhouette score.

    The silhouette score measures how well each node fits within its cluster
    compared to other clusters. Scores range from -1 to 1, where higher is better.
    """
    silhouette_scores = []

    for n_clusters in range(*n_clusters_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding)
        score = silhouette_score(embedding, cluster_labels)
        silhouette_scores.append((n_clusters, score))
        print(f"k={n_clusters}: silhouette score = {score:.3f}")

    # Select the number of clusters with highest silhouette score
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"\nOptimal number of clusters: {optimal_k}")

    # Perform final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    return kmeans.fit_predict(embedding)

# Find clusters
cluster_labels = find_optimal_clusters(embedding)
```

::: {.column-margin}
The silhouette score measures both cohesion (how close nodes are within their cluster) and separation (how far clusters are from each other).
:::

Now let's visualize the discovered clusters on the network:

```{python}
# Visualize the clustering results
cmap = sns.color_palette().as_hex()
igraph.plot(
    g,
    vertex_color=[cmap[label] for label in cluster_labels],
    bbox=(500, 500),
    vertex_size=20
)
```

The K-means algorithm successfully identifies community structure using only the learned embeddings, demonstrating that DeepWalk captures meaningful structural properties of the network.

### Node2vec: Flexible Graph Embeddings

Node2vec extends DeepWalk by introducing a biased random walk strategy. Instead of uniformly choosing the next node, node2vec uses two parameters, $p$ and $q$, to control the exploration strategy:

- **Return parameter $p$**: Controls the likelihood of returning to the previous node
- **In-out parameter $q$**: Controls whether the walk explores locally (BFS-like) or ventures further (DFS-like)

::: {.column-margin}
Node2vec was introduced by Grover and Leskovec (2016). The biased walk allows it to learn embeddings that capture different structural properties depending on the task.
:::

#### Step 1: Implement Biased Random Walk

The key innovation in node2vec is the biased random walk. Let's implement it step by step.
```{python}
def node2vec_random_walk(net, start_node, walk_length, p, q):
    """
    Generate a biased random walk for node2vec.

    Parameters:
    -----------
    net : sparse matrix
        Adjacency matrix of the network
    start_node : int
        Starting node for the walk
    walk_length : int
        Length of the walk
    p : float
        Return parameter (controls likelihood of returning to previous node)
    q : float
        In-out parameter (controls BFS vs DFS behavior)

    Returns:
    --------
    walk : list
        List of node indices representing the biased random walk
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(net[cur].indices)

        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                # First step: uniform random choice
                walk.append(np.random.choice(cur_nbrs))
            else:
                # Subsequent steps: biased choice based on p and q
                prev = walk[-2]
                next_node = biased_choice(net, cur_nbrs, prev, p, q)
                walk.append(next_node)
        else:
            break

    return walk

def biased_choice(net, neighbors, prev, p, q):
    """
    Choose the next node with bias controlled by p and q.

    The transition probability is:
    - 1/p if returning to the previous node
    - 1   if moving to a neighbor of the previous node (distance 1)
    - 1/q if moving away from the previous node (distance 2)
    """
    unnormalized_probs = []

    for neighbor in neighbors:
        if neighbor == prev:
            # Returning to previous node
            unnormalized_probs.append(1 / p)
        elif neighbor in net[prev].indices:
            # Moving to a common neighbor (BFS-like)
            unnormalized_probs.append(1.0)
        else:
            # Moving away from previous node (DFS-like)
            unnormalized_probs.append(1 / q)

    # Normalize probabilities
    norm_const = sum(unnormalized_probs)
    normalized_probs = [prob / norm_const for prob in unnormalized_probs]

    # Sample next node
    return np.random.choice(neighbors, p=normalized_probs)
```

::: {.column-margin}
**Understanding p and q:**

- Small $p$ ($p < 1$): Encourages returning to previous node, leading to local exploration
- Large $q$ ($q > 1$): Discourages moving away, resulting in BFS-like behavior
- Small $q$ ($q < 1$): Encourages exploration, resulting in DFS-like behavior
:::

#### Step 2: Generate Walks and Train Model

Now let's generate biased random walks and train the word2vec model. We'll use $p=1$ and $q=0.1$, which encourages outward exploration (DFS-like behavior) to capture community structure.

```{python}
# Generate biased random walks
p = 1.0   # Return parameter
q = 0.1   # In-out parameter (q < 1 means DFS-like)

walks_node2vec = []
for i in range(n_nodes):
    for _ in range(n_walkers_per_node):
        walks_node2vec.append(node2vec_random_walk(A, i, walk_length, p, q))

print(f"Generated {len(walks_node2vec)} biased random walks")
print(f"Example walk: {walks_node2vec[0][:10]}...")
```

::: {.column-margin}
With $q=0.1$, the walk is 10 times more likely to explore distant nodes than return to the immediate neighborhood, encouraging discovery of global community structure.
:::

Train the word2vec model on the biased walks:

```{python}
# Train node2vec model
model_node2vec = Word2Vec(
    walks_node2vec,
    vector_size=32,
    window=3,
    min_count=1,
    sg=1,
    hs=1
)

# Extract embeddings
embedding_node2vec = np.array([model_node2vec.wv[i] for i in range(n_nodes)])
print(f"Node2vec embedding shape: {embedding_node2vec.shape}")
```

#### Step 3: Visualize Node2vec Embeddings

Let's visualize the node2vec embeddings and compare them with DeepWalk.

```{python}
#| echo: false

# Reduce node2vec embeddings to 2D
reducer_n2v = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, metric="cosine")
xy_n2v = reducer_n2v.fit_transform(embedding_node2vec)

output_notebook()

degrees = A.sum(axis=1).A1

source_n2v = ColumnDataSource(data=dict(
    x=xy_n2v[:, 0],
    y=xy_n2v[:, 1],
    size=np.sqrt(degrees / np.max(degrees)) * 30,
    community=[palette[label] for label in g.vs["label"]],
    name = [str(i) for i in range(n_nodes)]
))

p_n2v = figure(title="Node2vec Embeddings (UMAP projection)", x_axis_label="UMAP 1", y_axis_label="UMAP 2")
p_n2v.scatter('x', 'y', size='size', source=source_n2v, line_color="black", color="community")

hover = HoverTool()
hover.tooltips = [
    ("Node", "@name"),
    ("Community", "@community")
]
p_n2v.add_tools(hover)

show(p_n2v)
```

Notice how the node2vec embeddings with $q=0.1$ (DFS-like exploration) create even more distinct community clusters compared to DeepWalk. This is because the biased walk explores the community structure more thoroughly.

#### Step 4: Clustering Analysis

Let's apply K-means clustering to the node2vec embeddings:

```{python}
# Find optimal clusters for node2vec embeddings
cluster_labels_n2v = find_optimal_clusters(embedding_node2vec)

# Visualize the clustering results
igraph.plot(
    g,
    vertex_color=[palette[label] for label in cluster_labels_n2v],
    bbox=(500, 500),
    vertex_size=20,
    vertex_label=[str(i) for i in range(n_nodes)]
)
```

By tuning the $p$ and $q$ parameters, node2vec can adapt to different network analysis tasks - from community detection (small $q$) to role discovery (large $q$).