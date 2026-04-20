"""
Benchmark: scalar per-classifier covers() vs vectorized approaches.

Models one explain_sample() call from the taxi experiment:
  - 4500 positive + 4500 negative training samples
  - 10 numeric features, 16 binary features
  - Query intersected with every training sample → hypothesis
  - Each hypothesis evaluated against both subsets (tp + fp)

Run from the repo root:
    python benchmarks/covers_vectorized.py
"""

import time
import numpy as np
import numba

# ---------------------------------------------------------------------------
# Synthetic data  (matches taxi profile)
# ---------------------------------------------------------------------------

N_POS = 4500
N_NEG = 4500
N_BINARY = 16
N_NUMERIC = 10
REPEATS = 7

rng = np.random.default_rng(42)

query_binary  = rng.integers(0, 2, (N_BINARY,),  dtype=bool)
query_numeric = rng.standard_normal((N_NUMERIC,))

# Training sets — stored F-order to match current Subset layout
pos_binary  = np.asfortranarray(rng.integers(0, 2, (N_POS, N_BINARY), dtype=bool))
pos_numeric = np.asfortranarray(rng.standard_normal((N_POS, N_NUMERIC)))
neg_binary  = np.asfortranarray(rng.integers(0, 2, (N_NEG, N_BINARY), dtype=bool))
neg_numeric = np.asfortranarray(rng.standard_normal((N_NEG, N_NUMERIC)))


# ---------------------------------------------------------------------------
# Scalar baseline — mirrors current covers() called once per Classifier
# ---------------------------------------------------------------------------

def scalar_tp_fp(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
):
    """Current approach: one loop iteration per training sample."""
    n = len(train_bin)
    tp = np.empty(n, dtype=np.int32)
    fp = np.empty(n, dtype=np.int32)

    for i in range(n):
        hyp_bin     = query_bin & train_bin[i]
        not_hyp_bin = ~hyp_bin
        hyp_min     = np.minimum(query_num, train_num[i])
        hyp_max     = np.maximum(query_num, train_num[i])

        cov_b = (sup_bin | not_hyp_bin).all(axis=1)
        cov_n = ((hyp_min <= sup_num) & (sup_num <= hyp_max)).all(axis=1)
        tp[i]  = (cov_b & cov_n).sum()

        cov_b = (opp_bin | not_hyp_bin).all(axis=1)
        cov_n = ((hyp_min <= opp_num) & (opp_num <= hyp_max)).all(axis=1)
        fp[i]  = (cov_b & cov_n).sum()

    return tp, fp


# ---------------------------------------------------------------------------
# Vectorized — chunked 3D broadcasting, binary + numeric together
# ---------------------------------------------------------------------------

def vectorized_chunked(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
    chunk: int = 128,
):
    """
    Precompute all hypothesis parameters, then evaluate covers in chunks.

    Chunk memory (binary ignored, numeric dominates):
        (chunk × max(N_POS, N_NEG) × N_NUMERIC) float64
        e.g. chunk=128 → 128 × 4500 × 10 × 8 B ≈ 46 MB
    """
    n = len(train_bin)

    # All hypothesis parameters at once
    not_hyp_bin = ~(query_bin & train_bin)          # (n, N_BINARY)
    hyp_min     = np.minimum(query_num, train_num)  # (n, N_NUMERIC)
    hyp_max     = np.maximum(query_num, train_num)  # (n, N_NUMERIC)

    tp = np.empty(n, dtype=np.int32)
    fp = np.empty(n, dtype=np.int32)

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))

        b  = not_hyp_bin[sl]   # (c, N_BINARY)
        lo = hyp_min[sl]       # (c, N_NUMERIC)
        hi = hyp_max[sl]       # (c, N_NUMERIC)

        # supporters
        cov_b = (b[:, None, :] | sup_bin[None, :, :]).all(axis=2)   # (c, N_POS)
        cov_n = ((lo[:, None, :] <= sup_num[None, :, :]) &
                 (sup_num[None, :, :] <= hi[:, None, :])).all(axis=2)
        tp[sl] = (cov_b & cov_n).sum(axis=1)

        # opposers
        cov_b = (b[:, None, :] | opp_bin[None, :, :]).all(axis=2)   # (c, N_NEG)
        cov_n = ((lo[:, None, :] <= opp_num[None, :, :]) &
                 (opp_num[None, :, :] <= hi[:, None, :])).all(axis=2)
        fp[sl] = (cov_b & cov_n).sum(axis=1)

    return tp, fp


# ---------------------------------------------------------------------------
# Vectorized — matmul for binary (BLAS SGEMM), chunked 3D for numeric
# ---------------------------------------------------------------------------

def vectorized_matmul_binary(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
    chunk: int = 128,
):
    """
    Binary covers via float32 matmul:
        covers_binary[i,j] = (hyp_bin[i] & ~subset_bin[j]).any() == False
                           = (hyp_bin @ ~subset_bin.T)[i,j] == 0
    Numeric covers via chunked 3D broadcasting.
    """
    n = len(train_bin)

    hyp_bin = query_bin & train_bin                 # (n, N_BINARY)
    hyp_min = np.minimum(query_num, train_num)      # (n, N_NUMERIC)
    hyp_max = np.maximum(query_num, train_num)      # (n, N_NUMERIC)

    # Binary covers matrices — one matmul each, O(n × N_BINARY × n_subset)
    hyp_f32      = hyp_bin.astype(np.float32)          # (n, N_BINARY)
    not_sup_f32  = (~sup_bin).astype(np.float32).T     # (N_BINARY, N_POS)  C-order
    not_opp_f32  = (~opp_bin).astype(np.float32).T     # (N_BINARY, N_NEG)

    cov_bin_sup = (hyp_f32 @ not_sup_f32) == 0         # (n, N_POS)  bool
    cov_bin_opp = (hyp_f32 @ not_opp_f32) == 0         # (n, N_NEG)  bool

    tp = np.empty(n, dtype=np.int32)
    fp = np.empty(n, dtype=np.int32)

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        lo = hyp_min[sl]   # (c, N_NUMERIC)
        hi = hyp_max[sl]   # (c, N_NUMERIC)

        cov_n_sup = ((lo[:, None, :] <= sup_num[None, :, :]) &
                     (sup_num[None, :, :] <= hi[:, None, :])).all(axis=2)
        tp[sl] = (cov_bin_sup[sl] & cov_n_sup).sum(axis=1)

        cov_n_opp = ((lo[:, None, :] <= opp_num[None, :, :]) &
                     (opp_num[None, :, :] <= hi[:, None, :])).all(axis=2)
        fp[sl] = (cov_bin_opp[sl] & cov_n_opp).sum(axis=1)

    return tp, fp


# ---------------------------------------------------------------------------
# Vectorized — feature-wise iteration (no 3D intermediate)
#
# Instead of (chunk, n_subset, n_features), accumulate (chunk, n_subset) bool
# by iterating over features.  Each feature touches one contiguous column of
# the F-order subset array → stays in L1/L2.
# ---------------------------------------------------------------------------

def vectorized_feature_iter(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
    chunk: int = 128,
):
    n = len(train_bin)

    hyp_bin = query_bin & train_bin                 # (n, N_BINARY)
    hyp_min = np.minimum(query_num, train_num)      # (n, N_NUMERIC)
    hyp_max = np.maximum(query_num, train_num)      # (n, N_NUMERIC)

    tp = np.empty(n, dtype=np.int32)
    fp = np.empty(n, dtype=np.int32)

    for start in range(0, n, chunk):
        sl  = slice(start, min(start + chunk, n))
        hb  = hyp_bin[sl]   # (c, N_BINARY)
        lo  = hyp_min[sl]   # (c, N_NUMERIC)
        hi  = hyp_max[sl]   # (c, N_NUMERIC)
        c   = hi.shape[0]

        # supporters ---
        cov = np.ones((c, len(sup_bin)), dtype=bool)
        for k in range(N_BINARY):
            # covers bit k: sup_bin[:, k] | ~hb[:, k]
            cov &= sup_bin[:, k] | ~hb[:, k:k+1]
        for k in range(N_NUMERIC):
            col = sup_num[:, k]   # (N_POS,) — contiguous (F-order)
            cov &= (lo[:, k:k+1] <= col) & (col <= hi[:, k:k+1])
        tp[sl] = cov.sum(axis=1)

        # opposers ---
        cov[:] = True
        for k in range(N_BINARY):
            cov &= opp_bin[:, k] | ~hb[:, k:k+1]
        for k in range(N_NUMERIC):
            col = opp_num[:, k]
            cov &= (lo[:, k:k+1] <= col) & (col <= hi[:, k:k+1])
        fp[sl] = cov.sum(axis=1)

    return tp, fp


# ---------------------------------------------------------------------------
# Numba — compiled triple loop, sequential and parallel
#
# Memory layout: C-order subsets so that subset[j] is a contiguous row,
# which is the access pattern of the innermost Numba loop (over features k).
# Early termination over features avoids wasted work when a sample is already
# known to be uncovered.
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _numba_covers_count(hyp_bin, hyp_min, hyp_max, sub_bin, sub_num):
    """Count how many subset rows are covered by a single hypothesis."""
    n_sub    = sub_bin.shape[0]
    n_binary = hyp_bin.shape[0]
    n_num    = hyp_min.shape[0]
    count = 0
    for j in range(n_sub):
        ok = True
        for k in range(n_binary):
            if hyp_bin[k] and not sub_bin[j, k]:
                ok = False
                break
        if not ok:
            continue
        for k in range(n_num):
            v = sub_num[j, k]
            if v < hyp_min[k] or v > hyp_max[k]:
                ok = False
                break
        if ok:
            count += 1
    return count


@numba.njit(cache=True)
def numba_sequential(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
):
    n = train_bin.shape[0]
    tp = np.empty(n, numba.int32)
    fp = np.empty(n, numba.int32)
    for i in range(n):
        hyp_bin = query_bin & train_bin[i]
        hyp_min = np.minimum(query_num, train_num[i])
        hyp_max = np.maximum(query_num, train_num[i])
        tp[i] = _numba_covers_count(hyp_bin, hyp_min, hyp_max, sup_bin, sup_num)
        fp[i] = _numba_covers_count(hyp_bin, hyp_min, hyp_max, opp_bin, opp_num)
    return tp, fp


@numba.njit(cache=True, parallel=True)
def numba_parallel(
    query_bin, query_num,
    train_bin, train_num,
    sup_bin, sup_num,
    opp_bin, opp_num,
):
    n = train_bin.shape[0]
    tp = np.empty(n, numba.int32)
    fp = np.empty(n, numba.int32)
    for i in numba.prange(n):
        hyp_bin = query_bin & train_bin[i]
        hyp_min = np.minimum(query_num, train_num[i])
        hyp_max = np.maximum(query_num, train_num[i])
        tp[i] = _numba_covers_count(hyp_bin, hyp_min, hyp_max, sup_bin, sup_num)
        fp[i] = _numba_covers_count(hyp_bin, hyp_min, hyp_max, opp_bin, opp_num)
    return tp, fp


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench(fn, *args, label, repeats=REPEATS, **kwargs):
    fn(*args, **kwargs)  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    mean_ms = 1000 * sum(times) / len(times)
    min_ms  = 1000 * min(times)
    print(f"  {label:<40}  mean {mean_ms:7.1f} ms   min {min_ms:7.1f} ms")
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"Data: {N_POS} pos + {N_NEG} neg training samples, "
      f"{N_BINARY} binary + {N_NUMERIC} numeric features")
print(f"Each approach computes tp+fp for all {N_POS + N_NEG} classifiers.")
print(f"({REPEATS} repeats, F-order subsets)\n")

args = (query_binary, query_numeric,
        pos_binary, pos_numeric,
        pos_binary, pos_numeric,   # supporters = positive subset
        neg_binary, neg_numeric)   # opposers   = negative subset

print("--- scalar (current) ---")
ref_tp, ref_fp = bench(scalar_tp_fp, *args, label="scalar")

print("\n--- vectorized chunked (binary + numeric together) ---")
for chunk in [32, 64, 128, 256, 512]:
    r_tp, r_fp = bench(
        vectorized_chunked, *args,
        chunk=chunk, label=f"chunk={chunk}",
    )
    assert np.array_equal(r_tp, ref_tp) and np.array_equal(r_fp, ref_fp), \
        f"Mismatch at chunk={chunk}"

print("\n--- vectorized matmul binary + chunked numeric ---")
for chunk in [32, 64, 128, 256, 512]:
    r_tp, r_fp = bench(
        vectorized_matmul_binary, *args,
        chunk=chunk, label=f"chunk={chunk}",
    )
    assert np.array_equal(r_tp, ref_tp) and np.array_equal(r_fp, ref_fp), \
        f"Mismatch at chunk={chunk}"

print("\n--- vectorized feature-wise iteration (no 3D intermediate) ---")
for chunk in [32, 64, 128, 256, 512]:
    r_tp, r_fp = bench(
        vectorized_feature_iter, *args,
        chunk=chunk, label=f"chunk={chunk}",
    )
    assert np.array_equal(r_tp, ref_tp) and np.array_equal(r_fp, ref_fp), \
        f"Mismatch at chunk={chunk}"

print("\n--- numba (C-order subsets; first call compiles) ---")
# C-order is optimal for Numba's row-access pattern (subset[j] is contiguous)
sup_bin_c  = np.ascontiguousarray(pos_binary)
sup_num_c  = np.ascontiguousarray(pos_numeric)
opp_bin_c  = np.ascontiguousarray(neg_binary)
opp_num_c  = np.ascontiguousarray(neg_numeric)
train_bin_c = np.ascontiguousarray(pos_binary)
train_num_c = np.ascontiguousarray(pos_numeric)

numba_args = (query_binary, query_numeric,
              train_bin_c, train_num_c,
              sup_bin_c, sup_num_c,
              opp_bin_c, opp_num_c)

print("  compiling...", end=" ", flush=True)
t0 = time.perf_counter()
numba_sequential(*numba_args)
numba_parallel(*numba_args)
print(f"done ({1000*(time.perf_counter()-t0):.0f} ms)")

r_tp, r_fp = bench(numba_sequential, *numba_args, label="sequential")
assert np.array_equal(r_tp, ref_tp) and np.array_equal(r_fp, ref_fp), "sequential mismatch"

r_tp, r_fp = bench(numba_parallel, *numba_args, label="parallel (prange)")
assert np.array_equal(r_tp, ref_tp) and np.array_equal(r_fp, ref_fp), "parallel mismatch"

print("\nAll results verified identical to scalar baseline.")
