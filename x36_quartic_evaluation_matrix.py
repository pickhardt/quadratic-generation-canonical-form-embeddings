#!/usr/bin/env python3
# Exact F_p evaluation matrix for quartic canonical-function products on X(3,6).
# Selects 18695 quartic monomials by deleting an invertible 81-column minor
# of the certified quartic residual matrix, then evaluates at deterministic points.
import gc,hashlib,itertools,json,os,random,re,sys,time
import numpy as np
from flint import nmod_mat
sys.path.insert(0,os.getcwd()); import y36_cubic_sieve as y
p=1000003; seed_points=314159265; t=time.time()
D=np.load('x36_cubic_basis_data.npz')
coord=list(map(int,D['coord'])); perm=list(map(int,D['perm'])); pivs=set(map(int,D['quadratic_pivs']))
standard3=[tuple(map(int,x)) for x in D['standard_cubics']]; cnonp=list(map(int,D['cubic_nonp']))
assert len(coord)==126 and len(pivs)==1385 and len(cnonp)==780
logical=y.grevlex_degree2_increasing(126)
mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
qleads={mons[j] for j in range(len(mons)) if j not in pivs}
cleads={standard3[j] for j in cnonp}
standard4=[]
for m in itertools.combinations_with_replacement(range(126),4):
 if any((m[i],m[j]) in qleads for i in range(4) for j in range(i+1,4)): continue
 if any(tuple(m[z] for z in range(4) if z!=omit) in cleads for omit in range(4)): continue
 standard4.append(m)
assert len(standard4)==18776
# Load the already-computed exact profile chunks. They cover coordinates 4184..18775
# and contain an invertible 81-column minor, so no unfinished profile chunks are needed.
# The exact quartic relation matrix is archived whole as
# x36_quartic_full_profiles.npz['relation_matrix'] (81 x 18776).  Columns
# 4184..18775 are the profile range the selection below scans.
COL0=4184
Rel=np.load('x36_quartic_full_profiles.npz')['relation_matrix'][:,COL0:18776]
assert Rel.shape==(81,14592)
# Greedily select independent columns, scanning from high monomial order downward.
basis={}; omit_rel=[]
for j in range(Rel.shape[1]-1,-1,-1):
 v=Rel[:,j].astype(np.int64)%p
 for k,b in basis.items():
  a=int(v[k])
  if a: v=(v-a*b)%p
 nz=np.flatnonzero(v)
 if len(nz):
  k=int(nz[0]); v=(v*pow(int(v[k]),p-2,p))%p
  basis[k]=v; omit_rel.append(j)
  if len(basis)==81: break
assert len(omit_rel)==81
omit=sorted(COL0+j for j in omit_rel)
Minor=Rel[:,omit_rel]
minor_det=int(nmod_mat(Minor.tolist(),p).det())
assert minor_det!=0
keep=[j for j in range(18776) if j not in set(omit)]
assert len(keep)==18695
selected=np.array([standard4[j] for j in keep],dtype=np.uint8)
monhash=hashlib.sha256(selected.tobytes()).hexdigest()
print(f'standard4=18776 omitted=81 minor_det={minor_det} selected=18695 monomial_sha256={monhash} elapsed={time.time()-t:.1f}s',flush=True)
del Rel,Minor,basis,D;gc.collect()
# Compile the 240 HPP canonical functions.
tails=[];seen=set()
for q in itertools.permutations(range(1,6)):
 if q in seen: continue
 tails.append(q);seen.add(q);seen.add(tuple(reversed(q)))
SIGMA=[(0,)+q for q in tails]
def compile_x(s):
 def P(tt): return y.signed_index(tuple(s[i-1] for i in tt))
 B=tuple(P(tt) for tt in [(1,2,3),(1,2,6),(1,4,5),(2,3,4),(3,5,6),(4,5,6)])
 C=(P((2,4,5)),tuple(P(tt) for tt in [(1,2,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,5,6),(4,5,6)]))
 q1=tuple(P(tt) for tt in [(1,2,3),(3,4,5),(1,5,6),(2,4,6)])
 q2=tuple(P(tt) for tt in [(2,3,4),(4,5,6),(1,2,6),(1,3,5)])
 DD=(q1,q2,tuple(P(tt) for tt in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
 cyc=tuple(y.signed_index((s[i],s[(i+1)%6],s[(i+2)%6])) for i in range(6))
 return cyc,B,C,DD
COMP=[compile_x(s) for s in SIGMA]
def all_forms(X):
 pl=[y.det3cols(X,z)%p for z in y.TRIPLES]; pairs=[]
 Ds=[];Cs=[];Bs=[];As=[]
 for A,B,C,DD in COMP:
  Ds.append(((y.prodP(pl,DD[0],p)-y.prodP(pl,DD[1],p))%p,y.prodP(pl,DD[2],p)))
  Cs.append((y.pv(pl,C[0],p),y.prodP(pl,C[1],p)))
  Bs.append((1,y.prodP(pl,B,p))); As.append((1,y.prodP(pl,A,p)))
 pairs=Ds+Cs+Bs+As; inv=y.batch_invert([b for a,b in pairs],p)
 return [(a*inv[i])%p for i,(a,b) in enumerate(pairs)]
rng=random.Random(seed_points)
def random_point():
 while True:
  X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
  try: return X,all_forms(X)
  except ZeroDivisionError: pass
nfull=18695; nrows=int(os.getenv('NROWS',str(nfull))); assert 1<=nrows<=nfull
outfile=os.getenv('OUTFILE','x36_quartic_eval_18695_u32.bin' if nrows==nfull else f'x36_quartic_eval_smoke_{nrows}_u32.bin')
idx=selected.astype(np.int64)
points=np.empty((nrows,3,6),dtype=np.uint32)
M=np.memmap(outfile,dtype='<u4',mode='w+',shape=(nrows,nfull))
print(f'evaluating exact matrix {nrows}x{nfull}, expected under 5 minutes; output={outfile}',flush=True)
for z in range(nrows):
 X,f=random_point(); points[z]=X
 a=np.asarray([f[i] for i in coord],dtype=np.uint64)
 v=(a[idx[:,0]]*a[idx[:,1]])%p
 v=(v*a[idx[:,2]])%p; v=(v*a[idx[:,3]])%p
 M[z,:]=v.astype(np.uint32)
 if (z+1)%500==0 or z+1==nrows: print(f'rows {z+1}/{nrows} elapsed={time.time()-t:.1f}s',flush=True)
M.flush(); del M;gc.collect()
ph=hashlib.sha256(points.astype('<u4').tobytes()).hexdigest()
np.save('x36_quartic_eval_points.npy',points)
h=hashlib.sha256()
with open(outfile,'rb') as ff:
 while True:
  b=ff.read(16<<20)
  if not b:break
  h.update(b)
res={'prime':p,'point_seed':seed_points,'rows':nrows,'cols':nfull,'matrix_file':outfile,'matrix_sha256':h.hexdigest(),'points_sha256':ph,'selected_monomials_sha256':monhash,'omitted_standard4_indices':omit,'quartic_relation_minor_det_mod_p':minor_det,'elapsed_seconds':time.time()-t}
open('x36_quartic_eval_matrix_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
