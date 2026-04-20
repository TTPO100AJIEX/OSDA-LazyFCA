import numba
import numpy as np


@numba.njit(cache=True)
def _covers_count(hyp_bin, hyp_min, hyp_max, sub_bin, sub_num):
    n_sub = sub_bin.shape[0]
    n_bin = hyp_bin.shape[0]
    n_num = hyp_min.shape[0]
    count = 0
    for j in range(n_sub):
        ok = True
        for k in range(n_bin):
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


@numba.njit(cache=True, parallel=True)
def compute_tp_fp(
    query_bin,
    query_num,
    train_bin,
    train_num,
    sup_bin,
    sup_num,
    opp_bin,
    opp_num,
):
    """
    For every training sample i, build hypothesis = query ∩ train[i] and
    count how many supporters (tp) and opposers (fp) it covers.
    All arrays must be C-order (row-major). The outer loop over training
    samples is parallelised across CPU cores with numba.prange.
    """
    n = train_bin.shape[0]
    tp = np.empty(n, numba.int32)
    fp = np.empty(n, numba.int32)
    for i in numba.prange(n):
        hyp_bin = query_bin & train_bin[i]
        hyp_min = np.minimum(query_num, train_num[i])
        hyp_max = np.maximum(query_num, train_num[i])
        tp[i] = _covers_count(hyp_bin, hyp_min, hyp_max, sup_bin, sup_num)
        fp[i] = _covers_count(hyp_bin, hyp_min, hyp_max, opp_bin, opp_num)
    return tp, fp
