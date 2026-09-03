#!/usr/bin/env python3
# Exact determinant over F_1000003 of the 23730x23730 Naruki quartic evaluation
# matrix.  A nonzero value certifies dim(S/I)_4 = 23730 = h^0(4L), i.e.
# surjectivity of Sym^4 H^0(L) -> H^0(4L) for Naruki's cross-ratio fourfold.
#
# Blocked LU over F_p using exact float64 BLAS; validated against
# python-flint nmod_mat.det() on random matrices and on submatrices of this
# very matrix (n = 600, 1500) before use.

import numpy as np

def det_mod_p(A, p, b=512, log=None):
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]; assert A.shape == (n, n)
    assert b * (p - 1) ** 2 < 2.0 ** 53, "block size too large for exact float64"
    det = 1; sign = 1
    for k in range(0, n, b):
        kb = min(b, n - k)
        # ---- panel factorization on columns k..k+kb (rows k..n) ----
        for j in range(k, k + kb):
            col = A[j:, j]
            nz = np.flatnonzero(col)
            if nz.size == 0:
                return 0
            piv = j + int(nz[0])
            if piv != j:
                A[[j, piv], :] = A[[piv, j], :]; sign = -sign
            pv = int(A[j, j]); det = det * pv % p
            inv = pow(pv, p - 2, p)
            below = A[j + 1:, j]
            np.mod(below * inv, p, out=below)
            if j + 1 < k + kb:
                blk = A[j + 1:, j + 1:k + kb]
                np.mod(blk - np.outer(below, A[j, j + 1:k + kb]), p, out=blk)
        if k + kb >= n:
            break
        # ---- U12 := L11^{-1} * A[k:k+kb, k+kb:] (unit lower triangular solve) ----
        U12 = A[k:k + kb, k + kb:]
        for j in range(kb - 1):
            lcol = A[k + j + 1:k + kb, k + j]
            np.mod(U12[j + 1:] - np.outer(lcol, U12[j]), p, out=U12[j + 1:])
        # ---- trailing update: A22 -= L21 @ U12 ----
        L21 = A[k + kb:, k:k + kb]
        T = A[k + kb:, k + kb:]
        for r0 in range(0, T.shape[0], 4096):          # chunked to bound temporaries
            r1 = min(r0 + 4096, T.shape[0])
            np.mod(T[r0:r1] - L21[r0:r1] @ U12, p, out=T[r0:r1])
        if log: log(k + kb, n)
    return (det if sign == 1 else (-det)) % p


import numpy as np, time, json, hashlib, sys

p=1000003; n=23730; t0=time.time()
MATRIX_FILE='y36_quartic_eval_23730_u32.bin'
EXPECTED_SHA='8641090a44c97a0ebf1111432d64fa26877b83cc84f26b5528ec6b7694ffca82'
_h=hashlib.sha256()
with open(MATRIX_FILE,'rb') as _f:
    for _b in iter(lambda:_f.read(16<<20),b''): _h.update(_b)
MATRIX_SHA=_h.hexdigest()
assert MATRIX_SHA==EXPECTED_SHA, f'matrix hash mismatch: {MATRIX_SHA} != {EXPECTED_SHA}'
print(f'matrix sha256 verified: {MATRIX_SHA}',flush=True)
M=np.memmap(MATRIX_FILE,dtype='<u4',mode='r',shape=(n,n))
print(f'loading {n}x{n} -> float64 ({n*n*8/2**30:.1f} GB)',flush=True)
A=np.empty((n,n),dtype=np.float64)
for r0 in range(0,n,2048):
    r1=min(r0+2048,n); A[r0:r1]=M[r0:r1]
del M
print(f'loaded elapsed={time.time()-t0:.1f}s; starting blocked LU (b=512)',flush=True)
last=[time.time()]
def log(done,total):
    now=time.time()
    if now-last[0]>30: print(f'  eliminated {done}/{total} elapsed={now-t0:.0f}s',flush=True); last[0]=now
d=det_mod_p(A,p,b=512,log=log)
el=time.time()-t0
print(f'DETERMINANT_MOD_P={d}',flush=True)
print(f'total_seconds={el:.1f}',flush=True)
json.dump({'prime':p,'n':n,'determinant_mod_p':d,'nonzero':bool(d),
           'matrix_sha256':MATRIX_SHA,
           'elapsed_seconds':el},open('y36_quartic_det_result.json','w'),indent=2,sort_keys=True)
