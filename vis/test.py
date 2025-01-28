import numpy as np
from scipy import sparse
from tqdm import tqdm
import numba
from numba import jit, float64, int64
from numba.experimental import jitclass

def calc_ppr_forward_push_fast(adj, source_nodes, alpha=0.15, epsilon=1e-6, batch_size=1000):
    """
    Compute Personalized PageRank (PPR) scores for specified source nodes using a numba-accelerated forward push method.

    This implementation uses an optimized forward push algorithm to efficiently compute PPR scores
    from given source nodes to all other nodes in the graph. It processes source nodes in batches
    for better performance and returns a sparse matrix containing significant PPR values.

    The forward push algorithm works by iteratively pushing residual probability mass from nodes
    to their neighbors until the residual at each node falls below an epsilon threshold relative
    to the node's degree.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix of the graph. Should be in CSR format for optimal performance.
        The matrix should be square and represent an unweighted graph.

    source_nodes : array-like, optional
        List or array of node indices from which to compute PPR scores.
        These are the starting points for the random walks.
        If None, PPR scores are computed for all nodes.

    alpha : float, optional (default=0.15)
        Teleportation probability, also known as the damping factor.
        Must be in the range (0, 1). Larger values give more weight to local graph structure.

    epsilon : float, optional (default=1e-6)
        Tolerance threshold for convergence. The algorithm stops pushing from a node when its
        residual falls below epsilon * node_degree. Smaller values give more accurate results
        but increase computation time.

    batch_size : int, optional (default=1000)
        Number of source nodes to process in parallel. Larger values may improve performance
        but require more memory.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix P where P[i,j] represents the PPR score for node j with respect to
        source node i. Only contains entries larger than epsilon. The number of rows equals
        the number of source nodes, and the number of columns equals the total number of nodes.

    Notes
    -----
    - The implementation uses Numba for performance optimization and requires the input
      adjacency matrix to be in CSR format.
    - Memory usage scales with batch_size and the number of significant PPR values
      (controlled by epsilon).
    - The algorithm automatically handles dangling nodes (nodes with no outgoing edges).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> # Create a small test graph
    >>> adj = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> source_nodes = [0, 1]  # Compute PPR only from nodes 0 and 1
    >>> ppr = calc_ppr_forward_push_fast(adj, source_nodes, alpha=0.15, epsilon=1e-6)
    >>> print(ppr.toarray())

    See Also
    --------
    _forward_push_single_source : Core implementation for single source node
    _forward_push_batch : Parallel implementation for multiple source nodes
    """

    n = adj.shape[0]
    adj_csr = adj.tocsr()
    if source_nodes is None:
        source_nodes = np.arange(n)
    source_nodes = np.array(source_nodes)
    n_sources = len(source_nodes)

    # Precompute values needed for all sources
    deg = np.array(adj_csr.sum(1)).flatten()
    deg_inv = 1.0 / np.maximum(deg, 1e-12)

    # Get CSR format arrays
    neighbors_indptr = adj_csr.indptr
    neighbors_indices = adj_csr.indices

    # Storage for sparse PPR matrix
    ppr_data = []
    ppr_rows = []
    ppr_cols = []

    # Process sources in batches
    for batch_start in tqdm(range(0, n_sources, batch_size), desc="Computing PPR"):
        batch_end = min(batch_start + batch_size, n_sources)
        batch_sources = source_nodes[batch_start:batch_end]

        # Compute PPR for batch
        for idx, source in enumerate(batch_sources):
            ppr = _forward_push_single_source(
                source, n, neighbors_indptr, neighbors_indices,
                deg_inv, deg, alpha, epsilon
            )

            # Store significant entries
            significant_indices = np.where(ppr > epsilon)[0]
            ppr_data.extend(ppr[significant_indices])
            ppr_rows.extend([batch_start + idx] * len(significant_indices))
            ppr_cols.extend(significant_indices)

    # Create sparse matrix
    return sparse.csr_matrix(
        (ppr_data, (ppr_rows, ppr_cols)),
        shape=(n_sources, n)
    )

@jit(nopython=True)
def _push_from_node(node, residual, ppr, neighbors_indptr, neighbors_indices,
                    deg_inv, alpha, epsilon, deg):
    """Optimized single node push operation"""
    push_value = residual[node]
    residual[node] = 0

    # Add to PPR
    ppr[node] += (1 - alpha) * push_value

    # Get node's neighbors using CSR format
    start = neighbors_indptr[node]
    end = neighbors_indptr[node + 1]

    # Only push if node has neighbors
    if start != end:
        # Push to neighbors
        neighbor_update = alpha * push_value * deg_inv[node]
        neighbors = neighbors_indices[start:end]

        # Update residuals and collect new active nodes
        new_active = []
        for neighbor in neighbors:
            old_residual = residual[neighbor]
            residual[neighbor] += neighbor_update

            # Check if node becomes active
            if old_residual <= epsilon * deg[neighbor] and residual[neighbor] > epsilon * deg[neighbor]:
                new_active.append(neighbor)

    return new_active

@jit(nopython=True)
def _forward_push_single_source(source, n, neighbors_indptr, neighbors_indices,
                              deg_inv, deg, alpha, epsilon):
    """Compute PPR scores for a single source node"""
    residual = np.zeros(n)
    ppr = np.zeros(n)

    # Initialize residual at source
    residual[source] = 1.0

    # Active nodes queue - implement as array for Numba compatibility
    active = np.zeros(n, dtype=np.int64)
    active[0] = source
    active_size = 1

    while active_size > 0:
        # Pop last active node
        active_size -= 1
        node = active[active_size]

        # Process node
        new_active = _push_from_node(node, residual, ppr, neighbors_indptr,
                                   neighbors_indices, deg_inv, alpha, epsilon, deg)

        # Add new active nodes
        for new_node in new_active:
            if active_size < n:  # Prevent overflow
                active[active_size] = new_node
                active_size += 1

    return ppr



# Additional utility for parallel processing multiple sources
@jit(nopython=True, parallel=True)
def _forward_push_batch(sources, n, neighbors_indptr, neighbors_indices,
                       deg_inv, deg, alpha, epsilon):
    """Process multiple source nodes in parallel"""
    batch_size = len(sources)
    results = np.zeros((batch_size, n))

    for i in numba.prange(batch_size):
        results[i] = _forward_push_single_source(
            sources[i], n, neighbors_indptr, neighbors_indices,
            deg_inv, deg, alpha, epsilon
        )

    return results