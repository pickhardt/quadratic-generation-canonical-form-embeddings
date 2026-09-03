#!/usr/bin/env python3
# Exact finite-field cubic interpolation artifact for HPP X(3,6).
# Reconstructs the 6616 quadratic rewrite rules and produces 780 cubic rules.
import gc,itertools,json,os,random,sys,time,hashlib
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat,nmod_mat
sys.path.insert(0,os.getcwd()); import y36_cubic_sieve as y
p=1000003; seed=260728368; rng=random.Random(seed); ctx=fmpz_mod_ctx(p); t=time.time()
# HPP permutations of labels 2,...,6 modulo reversal.
tails=[];seen=set()
for q in itertools.permutations(range(1,6)):
 if q in seen: continue
 tails.append(q);seen.add(q);seen.add(tuple(reversed(q)))
SIGMA=[(0,)+q for q in tails]
def compile_x(s):
 def P(t): return y.signed_index(tuple(s[i-1] for i in t))
 B=tuple(P(t) for t in [(1,2,3),(1,2,6),(1,4,5),(2,3,4),(3,5,6),(4,5,6)])
 C=(P((2,4,5)),tuple(P(t) for t in [(1,2,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,5,6),(4,5,6)]))
 q1=tuple(P(t) for t in [(1,2,3),(3,4,5),(1,5,6),(2,4,6)])
 q2=tuple(P(t) for t in [(2,3,4),(4,5,6),(1,2,6),(1,3,5)])
 D=(q1,q2,tuple(P(t) for t in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
 cyc=tuple(y.signed_index((s[i],s[(i+1)%6],s[(i+2)%6])) for i in range(6))
 return cyc,B,C,D
COMP=[compile_x(s) for s in SIGMA]
def all_forms(X):
 pl=[y.det3cols(X,z)%p for z in y.TRIPLES]; pairs=[]
 Ds=[];Cs=[];Bs=[];As=[]
 for A,B,C,D in COMP:
  Ds.append(((y.prodP(pl,D[0],p)-y.prodP(pl,D[1],p))%p,y.prodP(pl,D[2],p)))
  Cs.append((y.pv(pl,C[0],p),y.prodP(pl,C[1],p)))
  Bs.append((1,y.prodP(pl,B,p)));As.append((1,y.prodP(pl,A,p)))
 pairs=Ds+Cs+Bs+As; inv=y.batch_invert([b for a,b in pairs],p)
 return [(a*inv[i])%p for i,(a,b) in enumerate(pairs)]
def random_point():
 while True:
  X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
  try:return all_forms(X)
  except ZeroDivisionError:pass
print('Expected: quadratic reconstruction under 1 min; 6250x7030 cubic matrix and exact RREF a few minutes.',flush=True)
if os.getenv('SMOKE'):
 v=random_point(); assert len(v)==240; print('SMOKE_OK one exact canonical-form evaluation',flush=True); sys.exit(0)
coord=y.pivot_columns(fmpz_mod_mat([random_point() for _ in range(140)],ctx));assert len(coord)==126
perm=list(reversed(range(126))); logical=y.grevlex_degree2_increasing(126)
mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Y=[]
for z in range(1385):
 f=random_point();Y.append([f[i] for i in coord])
 if (z+1)%250==0:print(f'quadratic rows {z+1}/1385 elapsed={time.time()-t:.1f}s',flush=True)
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1385
pivs=[];last=-1
for row in range(rank):
 for j in range(last+1,R.ncols()):
  if int(R[row,j]):pivs.append(j);last=j;break
P=set(pivs);reducible={mons[j] for j in range(len(mons)) if j not in P};del R;gc.collect();assert len(reducible)==6616
standard=[c for c in itertools.combinations_with_replacement(range(126),3)
 if all(q not in reducible for q in ((c[0],c[1]),(c[0],c[2]),(c[1],c[2])))]
assert len(standard)==7030
print(f'quadratic done; standard cubics={len(standard)} elapsed={time.time()-t:.1f}s',flush=True)
rows=[];nrows=6250
for z in range(nrows):
 f=random_point();a=[f[i] for i in coord]
 rows.append([(a[i]*a[j]%p)*a[k]%p for i,j,k in standard])
 if (z+1)%250==0:print(f'cubic evaluation rows {z+1}/{nrows} elapsed={time.time()-t:.0f}s',flush=True)
print(f'constructing nmod_mat elapsed={time.time()-t:.0f}s',flush=True)
C=nmod_mat(rows,p);del rows;gc.collect();print('exact cubic rref begins',flush=True)
CR,crank=C.rref();del C;gc.collect();print(f'cubic rank={crank} elapsed={time.time()-t:.0f}s',flush=True);assert crank==6250
cpivs=[];last=-1
for row in range(crank):
 for j in range(last+1,CR.ncols()):
  if int(CR[row,j]):cpivs.append(j);last=j;break
CP=set(cpivs);nonp=[j for j in range(len(standard)) if j not in CP];assert len(nonp)==780
print('extracting 780x6250 cubic rewrite tails',flush=True)
tailsA=np.empty((len(nonp),crank),dtype=np.uint32)
for z,j in enumerate(nonp):
 for i in range(crank):tailsA[z,i]=int(CR[i,j])
 if (z+1)%50==0:print(f'tails {z+1}/780 elapsed={time.time()-t:.0f}s',flush=True)
del CR;gc.collect()
np.savez_compressed('x36_cubic_basis_data.npz',coord=np.array(coord,dtype=np.uint32),perm=np.array(perm,dtype=np.uint32),quadratic_pivs=np.array(pivs,dtype=np.uint32),standard_cubics=np.array(standard,dtype=np.uint8),cubic_pivs=np.array(cpivs,dtype=np.uint32),cubic_nonp=np.array(nonp,dtype=np.uint32),cubic_tails=tailsA)
def sha(fn):return hashlib.sha256(open(fn,'rb').read()).hexdigest()
res={'prime':p,'seed':seed,'quadratic_rank':rank,'quadratic_standard_cubics':len(standard),'cubic_rank':crank,'cubic_kernel':len(nonp),'artifact_sha256':sha('x36_cubic_basis_data.npz'),'elapsed_seconds':time.time()-t}
open('x36_cubic_basis_result.json','w').write(json.dumps(res,indent=2,sort_keys=True));print(json.dumps(res,indent=2,sort_keys=True),flush=True)
