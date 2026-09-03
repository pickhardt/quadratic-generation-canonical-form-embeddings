#!/usr/bin/env python3
# Deterministic finite-field cubic sieve for HPP's Y(3,6) canonical forms.
# Phase 1: select 150 canonical coordinates and construct the degree-2
# evaluation matrix. Phase 2: compute its column-rank profile in a genuine
# graded reverse lexicographic order and count the cubic shadow of the
# resulting quadratic leading monomials.

import argparse, hashlib, itertools, json, random, struct, sys, time
from flint import fmpz_mod_ctx, fmpz_mod_mat

TRIPLES=list(itertools.combinations(range(6),3))
TINDEX={t:i for i,t in enumerate(TRIPLES)}
PERMS=list(itertools.permutations(range(6)))

def det3cols(X, c):
    a,b,d=c
    return (X[0][a]*(X[1][b]*X[2][d]-X[1][d]*X[2][b])
           -X[0][b]*(X[1][a]*X[2][d]-X[1][d]*X[2][a])
           +X[0][d]*(X[1][a]*X[2][b]-X[1][b]*X[2][a]))

def signed_index(t):
    inv=((t[0]>t[1])+(t[0]>t[2])+(t[1]>t[2]))
    return (-1 if inv&1 else 1), TINDEX[tuple(sorted(t))]

# For every permutation, compile each ordered minor to (sign,index).
def compile_perm(s):
    def P(t): return signed_index(tuple(s[i-1] for i in t))
    q1=tuple(P(t) for t in [(1,2,3),(3,4,5),(1,5,6),(2,4,6)])
    q2=tuple(P(t) for t in [(2,3,4),(4,5,6),(1,2,6),(1,3,5)])
    A=(P((1,3,5)), q1,q2, tuple(P(t) for t in [(1,2,3),(3,4,5),(1,5,6)]))
    B=tuple(P(t) for t in [(1,2,3),(1,2,6),(1,4,5),(2,3,4),(3,5,6),(4,5,6)])
    C=(P((2,4,5)),tuple(P(t) for t in [(1,2,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,5,6),(4,5,6)]))
    D=(q1,q2,tuple(P(t) for t in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
    return A,B,C,D
COMPILED=[compile_perm(s) for s in PERMS]

def pv(pl, si, p):
    s,i=si
    return pl[i] if s==1 else (-pl[i])%p

def prodP(pl, seq,p):
    z=1
    for si in seq: z=(z*pv(pl,si,p))%p
    return z

def batch_invert(vals,p):
    n=len(vals); pref=[1]*(n+1)
    for i,x in enumerate(vals):
        if x==0: raise ZeroDivisionError
        pref[i+1]=(pref[i]*x)%p
    z=pow(pref[n],p-2,p); out=[0]*n
    for i in range(n-1,-1,-1):
        out[i]=(z*pref[i])%p; z=(z*vals[i])%p
    return out

def all_forms(X,p):
    pl=[det3cols(X,t)%p for t in TRIPLES]
    nums=[]; dens=[]
    # HPP order imagesD|imagesC|imagesB|imagesA
    Ds=[]; Cs=[]; Bs=[]; As=[]
    for A,B,C,D in COMPILED:
        q=(prodP(pl,D[0],p)-prodP(pl,D[1],p))%p
        Ds.append((q,prodP(pl,D[2],p)))
        Cs.append((pv(pl,C[0],p),prodP(pl,C[1],p)))
        Bs.append((1,prodP(pl,B,p)))
        # q sequences are same as D
        As.append((pv(pl,A[0],p),(q*prodP(pl,A[3],p))%p))
    pairs=Ds+Cs+Bs+As
    inv=batch_invert([b for a,b in pairs],p)
    return [(pairs[i][0]*inv[i])%p for i in range(2880)]

def random_point(rng,p):
    while True:
        X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
        try: return all_forms(X,p)
        except ZeroDivisionError: pass

def pivot_columns(A):
    R,rank=A.rref()
    piv=[]; last=-1
    # Each nonzero RREF row has its first nonzero in its pivot column.
    for i in range(rank):
        for j in range(last+1,R.ncols()):
            if int(R[i,j])!=0:
                piv.append(j); last=j; break
    if len(piv)!=rank: raise RuntimeError('pivot extraction failed')
    return piv

def grevlex_degree2_increasing(n):
    # Compare exponent vectors alpha,beta in increasing grevlex: at the last
    # differing variable, alpha is larger iff alpha is smaller in grevlex.
    mons=[(i,j) for i in range(n) for j in range(i,n)]
    def key(ij):
        e=[0]*n; e[ij[0]]+=1; e[ij[1]]+=1
        return tuple(reversed(e)) # smaller key means larger grevlex monomial
    return list(reversed(sorted(mons,key=key)))

def cubic_standard_count(pivot_pairs,n):
    P=set(pivot_pairs); count=0
    for i in range(n):
      for j in range(i,n):
       if (i,j) not in P: continue
       for k in range(j,n):
        if (i,k) in P and (j,k) in P: count+=1
    return count

def sha_ints(vals):
    h=hashlib.sha256()
    for x in vals: h.update(struct.pack('<I',x))
    return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--prime',type=int,default=1000003)
 ap.add_argument('--seed',type=int,default=260728368)
 ap.add_argument('--linear-points',type=int,default=170)
 ap.add_argument('--quad-points',type=int,default=1720)
 ap.add_argument('--smoke',action='store_true')
 args=ap.parse_args(); p=args.prime; rng=random.Random(args.seed); ctx=fmpz_mod_ctx(p)
 t=time.time()
 lp=30 if args.smoke else args.linear_points
 rows=[random_point(rng,p) for _ in range(lp)]
 A=fmpz_mod_mat(rows,ctx); piv=pivot_columns(A)
 print(f'linear_points={lp} linear_rank={len(piv)} elapsed={time.time()-t:.2f}',flush=True)
 if args.smoke: return
 if len(piv)!=150: raise RuntimeError(f'expected linear rank 150, got {len(piv)}')
 coord=piv
 print('coordinate_indices_sha256='+sha_ints(coord),flush=True)
 mons=grevlex_degree2_increasing(150)
 qp=args.quad_points; qrows=[]
 print(f'building degree-2 evaluation matrix {qp}x{len(mons)}',flush=True)
 for z in range(qp):
    f=random_point(rng,p); y=[f[i] for i in coord]
    qrows.append([(y[i]*y[j])%p for i,j in mons])
    if (z+1)%100==0: print(f'points={z+1}/{qp} elapsed={time.time()-t:.1f}',flush=True)
 print('constructing flint matrix',flush=True)
 Q=fmpz_mod_mat(qrows,ctx); del qrows
 print(f'rref start elapsed={time.time()-t:.1f}',flush=True)
 qpiv=pivot_columns(Q)
 print(f'quadratic_rank={len(qpiv)} elapsed={time.time()-t:.1f}',flush=True)
 pp=[mons[i] for i in qpiv]
 c=cubic_standard_count(pp,150)
 result={'prime':p,'seed':args.seed,'linear_points':lp,'quad_points':qp,
   'linear_rank':len(coord),'quadratic_rank':len(qpiv),
   'quadratic_kernel_dimension':len(mons)-len(qpiv),
   'quadratic_pivot_indices_sha256':sha_ints(qpiv),
   'cubic_standard_monomials':c,
   'cubic_shadow_rank_lower_bound':573800-c,
   'target_standard':7876,'target_shadow':565924}
 print(json.dumps(result,sort_keys=True,indent=2),flush=True)

if __name__=='__main__': main()
