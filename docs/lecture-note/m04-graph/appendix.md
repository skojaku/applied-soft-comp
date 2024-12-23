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

# Appendix

## Spectral Embedding with the Adjacency Matrix

The spectral embedding with the adjacency matrix is given by the following optimization problem:

$$
\min_{\mathbf{U}} J(\mathbf{U}),\quad J(\mathbf{U}) = \| \mathbf{A} - \mathbf{U}\mathbf{U}^\top \|_F^2
$$

We will approach the solution step by step based on the following steps:

1. We start taking a derivative of $J(\mathbf{U})$  with respect to $\mathbf{U}$.
2. We then set the derivative to zero (i.e., $\nabla J(\mathbf{U}) = 0$) and solve for $\mathbf{U}$.

1. Expand the Frobenius norm:

   The Frobenius norm for any matrix $\mathbf{M}$ is defined as:

   $\|\mathbf{M}\|_F^2 = \sum_{i,j} M_{ij}^2 = \text{Tr}(\mathbf{M}\mathbf{M}^\top)$

   Applying this to our problem:

   $J(\mathbf{U}) = \|\mathbf{A} - \mathbf{U}\mathbf{U}^\top\|_F^2 = \text{Tr}[(\mathbf{A} - \mathbf{U}\mathbf{U}^\top)(\mathbf{A} - \mathbf{U}\mathbf{U}^\top)^\top]$

   Expanding this:

   $= \text{Tr}(\mathbf{A}\mathbf{A}^\top - 2\mathbf{A}\mathbf{U}\mathbf{U}^\top + \mathbf{U}\mathbf{U}^\top\mathbf{U}\mathbf{U}^\top)$

2. Take the derivative with respect to $\mathbf{U}$:

   Using matrix calculus rules:

   $\frac{\partial \text{Tr}(\mathbf{A}\mathbf{A}^\top)}{\partial \mathbf{U}} = 0$

   $\frac{\partial \text{Tr}(\mathbf{A}\mathbf{U}\mathbf{U}^\top)}{\partial \mathbf{U}} = 2\mathbf{A}\mathbf{U}$

   $\frac{\partial \text{Tr}(\mathbf{U}\mathbf{U}^\top\mathbf{U}\mathbf{U}^\top)}{\partial \mathbf{U}} = 4\mathbf{U}\mathbf{U}^\top\mathbf{U}$

   Combining these:

   $\frac{\partial J}{\partial \mathbf{U}} = -4\mathbf{A}\mathbf{U} + 4\mathbf{U}\mathbf{U}^\top\mathbf{U}$

   Simplifying:

   $\frac{\partial J}{\partial \mathbf{U}} = -2\mathbf{A}\mathbf{U} + 2\mathbf{U}\mathbf{U}^\top\mathbf{U}$

3. Set the derivative to zero and solve:

   $-2\mathbf{A}\mathbf{U} + 2\mathbf{U}\mathbf{U}^\top\mathbf{U} = 0$

   $\mathbf{A}\mathbf{U} = \mathbf{U}\mathbf{U}^\top\mathbf{U}$

4. This equation is satisfied when $\mathbf{U}$ consists of eigenvectors of $\mathbf{A}$:

   Assume $\mathbf{U}$ consists of eigenvectors of $\mathbf{A}$:

   $\mathbf{A}\mathbf{U} = \mathbf{U}\mathbf{\Lambda}$

   where $\mathbf{\Lambda}$ is a diagonal matrix of eigenvalues.

   Since eigenvectors are orthonormal:

   $\mathbf{U}^\top\mathbf{U} = \mathbf{I}$

   Therefore:

   $\mathbf{U}\mathbf{U}^\top\mathbf{U} = \mathbf{U}$

   This shows our equation is satisfied when $\mathbf{U}$ consists of eigenvectors of $\mathbf{A}$.

5. To minimize $J(\mathbf{U})$, choose the eigenvectors corresponding to the $d$ largest eigenvalues.

   To understand why, consider the trace of our objective function:

   $J(\mathbf{U}) = \text{Tr}(\mathbf{A}\mathbf{A}^\top) - 2\text{Tr}(\mathbf{A}\mathbf{U}\mathbf{U}^\top) + \text{Tr}(\mathbf{U}\mathbf{U}^\top\mathbf{U}\mathbf{U}^\top)$

   Since $\mathbf{U}$ is orthogonal ($\mathbf{U}^\top\mathbf{U} = \mathbf{I}$), and trace is invariant under cyclic permutations, we can simplify:

   $J(\mathbf{U}) = \text{Tr}(\mathbf{A}\mathbf{A}^\top) - \text{Tr}(\mathbf{U}^\top\mathbf{A}\mathbf{U})$

   Let $\mathbf{U} = [\mathbf{u}_1, ..., \mathbf{u}_d]$ be the eigenvectors of $\mathbf{A}$ with corresponding eigenvalues $\lambda_1 \geq ... \geq \lambda_d$. Then:

   $\text{Tr}(\mathbf{U}^\top\mathbf{A}\mathbf{U}) = \sum_{i=1}^d \lambda_i$

   To minimize $J(\mathbf{U})$, maximize $\sum_{i=1}^d \lambda_i$ by selecting the eigenvectors corresponding to the $d$ largest eigenvalues.

The result is the collection of the $d$ eigenvectors corresponding to the $d$ largest eigenvalues, and it is one form of the spectral embedding.


## The proof of the Laplacian Eigenmap

The Laplacian Eigenmap is given by the following optimization problem:

$$
J_{LE}(\mathbf{U}) = \text{Tr}(\mathbf{U}^\top \mathbf{L} \mathbf{U})
$$

The step where we rewrite $J_{LE}(\mathbf{U})$ as $\text{Tr}(\mathbf{U}^\top \mathbf{L} \mathbf{U})$ is crucial for leveraging matrix derivatives. Let's break down this transformation step by step:

1. First, we rewrite $\mathbf{U}$ by column vectors:

   $$
   \mathbf{U} =
   \begin{bmatrix}
   \vert & \vert & & \vert \\
   \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_d \\
   \vert & \vert & & \vert
   \end{bmatrix}
   $$

   where $\mathbf{x}_i$ is the $i$-th column of $\mathbf{U}$.

2. We can expand the loss function $J_{LE}(\mathbf{U})$:

   $$
   J_{LE}(\mathbf{U}) = \sum_{i} \sum_{j} L_{ij} u_i^\top u_j = \sum_{i} \sum_{j} \sum_{d'} L_{ij} u_{i,d'} u_{j,d'}
   $$

3. Rearranging the order of summation:

   $$
   J_{LE}(\mathbf{U}) = \sum_{d'} \sum_{i} \sum_{j} L_{ij} u_{i,d'} u_{j,d'}
   $$

4. We can rewrite this as a matrix multiplication for each $d'$:

   $$
   J_{LE}(\mathbf{U}) = \sum_{d'} \mathbf{x}_{d'}^\top \mathbf{L} \mathbf{x}_{d'}
   $$

   where $\mathbf{x}_{d'}$ is the $d'$-th column of $\mathbf{U}$.

5. Finally, we can express this as a trace:

   $$
   J_{LE}(\mathbf{U}) = \text{Tr}(\mathbf{U}^\top \mathbf{L} \mathbf{U})
   $$

This final form expresses our objective function in terms of matrix operations, which allows us to use matrix calculus to find the optimal solution. The trace representation is a useful technique to leverage matrix calculus.