#!/usr/bin/env python3
# Dense exact interpolation of the 8582 quadratic-normal cubic monomials.
# Produces a 7876-element cubic basis and 706 cubic rewrite rules over F_1000003.
import gc,itertools,json,os,random,sys,time,hashlib,struct
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat,nmod_mat
sys.path.insert(0,os.getcwd())
import y36_cubic_sieve as y
p=1000003; seed=260728368; rng=random.Random(seed); ctx=fmpz_mod_ctx(p); t=time.time()
print('Expected: build/rref quadratic (~25s), build 7876x8582 cubic matrix (~3 min), exact nmod rref.',flush=True)
coord=y.pivot_columns(fmpz_mod_mat([y.random_point(rng,p) for _ in range(170)],ctx)); assert len(coord)==150
Y=[]
for z in range(1720):
 f=y.random_point(rng,p);Y.append([f[i] for i in coord])
perm=list(reversed(range(150))); logical=y.grevlex_degree2_increasing(150)
mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx); del Y
R,rank=Q.rref();del Q;assert rank==1720
pivs=[];last=-1
for row in range(rank):
 for j in range(last+1,R.ncols()):
  if int(R[row,j]):pivs.append(j);last=j;break
P=set(pivs); reducible={mons[j] for j in range(len(mons)) if j not in P}; del R;gc.collect()
standard=[c for c in itertools.combinations_with_replacement(range(150),3)
          if all(tuple(sorted(q)) not in reducible for q in [(c[0],c[1]),(c[0],c[2]),(c[1],c[2])])]
assert len(standard)==8582
print(f'quadratic done; standard cubics={len(standard)} elapsed={time.time()-t:.1f}s',flush=True)
rows=[]; nrows=7876
for z in range(nrows):
 f=y.random_point(rng,p); a=[f[i] for i in coord]
 rows.append([(a[i]*a[j]%p)*a[k]%p for i,j,k in standard])
 if (z+1)%250==0:print(f'cubic evaluation rows {z+1}/{nrows} elapsed={time.time()-t:.0f}s',flush=True)
print(f'constructing nmod_mat; Python rows resident elapsed={time.time()-t:.0f}s',flush=True)
C=nmod_mat(rows,p);del rows;gc.collect()
print(f'exact cubic rref begins elapsed={time.time()-t:.0f}s',flush=True)
CR,crank=C.rref();del C;gc.collect();print(f'cubic_rank={crank} elapsed={time.time()-t:.0f}s',flush=True)
assert crank==7876
cpivs=[];last=-1
for row in range(crank):
 for j in range(last+1,CR.ncols()):
  if int(CR[row,j]):cpivs.append(j);last=j;break
assert len(cpivs)==crank
CP=set(cpivs);nonp=[j for j in range(len(standard)) if j not in CP];assert len(nonp)==706
print('extracting 706x7876 rewrite tails',flush=True)
tails=np.empty((len(nonp),crank),dtype=np.uint32)
for z,j in enumerate(nonp):
 for i in range(crank):tails[z,i]=int(CR[i,j])
 if (z+1)%50==0:print(f'tails {z+1}/706 elapsed={time.time()-t:.0f}s',flush=True)
del CR;gc.collect()
np.savez_compressed('y36_cubic_basis_data.npz',coord=np.array(coord,dtype=np.uint32),perm=np.array(perm,dtype=np.uint32),quadratic_pivs=np.array(pivs,dtype=np.uint32),standard_cubics=np.array(standard,dtype=np.uint8),cubic_pivs=np.array(cpivs,dtype=np.uint32),cubic_nonp=np.array(nonp,dtype=np.uint32),cubic_tails=tails)
def sha_file(fn):
 h=hashlib.sha256()
 with open(fn,'rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
res={'prime':p,'seed':seed,'quadratic_rank':rank,'quadratic_standard_cubics':len(standard),'cubic_rank':crank,'cubic_kernel':len(nonp),'standard_cubic_basis_sha256':y.sha_ints(cpivs),'cubic_nonpivot_sha256':y.sha_ints(nonp),'artifact_sha256':sha_file('y36_cubic_basis_data.npz'),'elapsed_seconds':time.time()-t}
open('y36_cubic_basis_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
