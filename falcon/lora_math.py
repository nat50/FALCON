"""Core LoRA aggregation math (pure NumPy, framework-independent).

  - consensus input subspace V from ALL client A matrices
  - weighted full-update average from B-clients, projected onto V
  - factorization back into a global (A_global, B_global) pair
  - per-client rank truncation and optional B re-alignment

Shape conventions for one LoRA layer:
  A : (r, k)   rows live in the input space  R^k
  B : (d, r)   columns live in the output space  R^d
  dW = B @ A : (d, k)
"""

from typing import List, Tuple

import numpy as np


def consensus_subspace(a_matrices: List[np.ndarray], rank: int) -> np.ndarray:
    """Build an orthonormal basis V (R, k) of the shared input subspace.

    Stacks every client's A vertically (they all share k columns) and keeps the
    top-`rank` right singular vectors.
    """
    stacked = np.concatenate(a_matrices, axis=0)  # (sum_r, k)
    keep = min(rank, stacked.shape[0], stacked.shape[1])
    _, _, vt = np.linalg.svd(stacked, full_matrices=False)
    return vt[:keep, :]  # (R, k), rows orthonormal


def projector(v_basis: np.ndarray) -> np.ndarray:
    """Return the k x k projector P = V^T V onto the rows of V."""
    return v_basis.T @ v_basis


def alignment_score(a_matrix: np.ndarray, proj: np.ndarray) -> float:
    """Fraction of A's energy that lies inside the consensus subspace, in [0, 1].

    A high score means the client's directions agree with the consensus
    (shared knowledge); a low score means the client is idiosyncratic (private).
    """
    total = float(np.sum(a_matrix * a_matrix))
    if total <= 1e-12:
        return 0.0
    inside = float(np.sum((a_matrix @ proj) * a_matrix))
    return max(0.0, min(1.0, inside / total))


def _minmax(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] with the epsilon from the algorithm note."""
    eps = 1e-6
    return (values - np.min(values)) / (np.max(values) - np.min(values) + eps)


def compute_rank_scores(
    n_samples: List[float],
    loss_before: List[float],
    loss_after: List[float],
    align_scores: List[float],
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Compute dynamic rank scores S_i from data, learning difficulty, and novelty."""
    n_arr = np.asarray(n_samples, dtype=np.float64)
    before = np.asarray(loss_before, dtype=np.float64)
    after = np.asarray(loss_after, dtype=np.float64)
    align = np.asarray(align_scores, dtype=np.float64)

    eps = 1e-6
    data_score = _minmax(n_arr)
    progress = (before - after) / (before + eps)
    learn_score = 1.0 - _minmax(progress)
    novelty_score = 1.0 - np.clip(align, 0.0, 1.0)

    scores = alpha * data_score + beta * learn_score + gamma * novelty_score
    return np.clip(scores, 0.0, 1.0)


def allocate_ranks(rank_scores: List[float], rank_pool: List[int]) -> List[int]:
    """Map rank scores in [0, 1] to discrete ranks from the configured pool."""
    if not rank_pool:
        raise ValueError("rank_pool must not be empty")
    pool = sorted(int(rank) for rank in rank_pool)
    scores = np.clip(np.asarray(rank_scores, dtype=np.float64), 0.0, 1.0)
    indices = np.minimum((scores * len(pool)).astype(int), len(pool) - 1)
    return [pool[int(index)] for index in indices]


def factorize(delta_w: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a (d, k) update into (A_global (R, k), B_global (d, R)) via SVD.

    The singular values are split evenly between the two factors so that
    B_global @ A_global reconstructs delta_w.
    """
    u, s, vt = np.linalg.svd(delta_w, full_matrices=False)
    keep = min(rank, len(s))
    sqrt_s = np.sqrt(s[:keep])
    a_global = sqrt_s[:, None] * vt[:keep, :]      # (R, k)
    b_global = u[:, :keep] * sqrt_s[None, :]       # (d, R)
    return a_global, b_global


def merge_layer(
    a_all: List[np.ndarray],
    b_clients: List[np.ndarray],
    a_clients: List[np.ndarray],
    n_samples: List[float],
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate one LoRA layer across clients with heterogeneous ranks.

    Args:
        a_all: A matrices from ALL clients (used for the consensus subspace).
        b_clients: B matrices from the clients that uploaded B.
        a_clients: matching A matrices for those same clients (same order as b_clients).
        n_samples: sample counts for B-clients.
        rank: global rank R to keep.

    Returns:
        (A_global (R, k), B_global (d, R)).
    """
    v_basis = consensus_subspace(a_all, rank)
    proj = projector(v_basis)

    d_out = b_clients[0].shape[0]
    k_in = a_all[0].shape[1]
    delta_bar = np.zeros((d_out, k_in), dtype=np.float64)
    samples = np.asarray(n_samples, dtype=np.float64)
    weight_sum = float(np.sum(samples)) or 1.0
    for b_mat, a_mat, w in zip(b_clients, a_clients, samples):
        delta_bar += (w / weight_sum) * (b_mat @ a_mat)

    delta_shared = delta_bar @ proj  # keep only consensus directions
    return factorize(delta_shared, rank)


def truncate_factors(
    a_global: np.ndarray, b_global: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the global factors down to a client's own rank (top components)."""
    return a_global[:rank, :].copy(), b_global[:, :rank].copy()


def realign_B(
    b_old: np.ndarray, a_old: np.ndarray, a_new: np.ndarray
) -> np.ndarray:
    """Re-align a personal B so the client's function is preserved when A changes.

    Finds M (least squares) with a_new ~= M @ a_old, then returns b_old @ pinv(M)
    so that (b_old @ pinv(M)) @ a_new ~= b_old @ a_old.
    """
    m, _, _, _ = np.linalg.lstsq(a_old.T, a_new.T, rcond=None)  # a_old^T M^T = a_new^T
    m = m.T  # (r_new, r_old)
    return b_old @ np.linalg.pinv(m)
