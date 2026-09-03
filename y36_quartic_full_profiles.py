#!/usr/bin/env python3
# Complete the exact standard-quartic coordinate matrix of the durable 87 raw
# ambiguity relations.  The existing artifact stores coordinates 23816..8073;
# this computes 8072..0 in descending order and writes the full matrix in the
# natural standard4 enumeration.  Arithmetic is over F_1000003.
import gc,itertools,json,os,random,sys,time,hashlib
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd()); import y36_cubic_sieve as y
p=1000003; seed=260728368; ctx=fmpz_mod_ctx(p); rng=random.Random(seed); t=time.time()
chunk=int(os.getenv('CHUNK','256')); maxchunks=int(os.getenv('MAXCHUNKS','0'))
D=np.load('y36_cubic_basis_data.npz'); coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
print(f'Expected: reconstruction <1 min; lower-coordinate profiling roughly 5-10 min. chunk={chunk} maxchunks={maxchunks}',flush=True)
coord2=y.pivot_columns(fmpz_mod_mat([y.random_point(rng,p) for _ in range(170)],ctx)); assert coord2==coord
Y=[]
for z in range(1720):
 f=y.random_point(rng,p);Y.append([f[i] for i in coord])
logical=y.grevlex_degree2_increasing(150);mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1720
P=set(pivs); qrules={}
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 qrules[m]=rr
del R;gc.collect(); assert len(qrules)==9605
crules={}
for z,j in enumerate(cnonp):crules[standard3[j]]=(ctails[z],z)
def qdivs4(m):
 ans=[]
 for ij in itertools.combinations(range(4),2):
  q=(m[ij[0]],m[ij[1]])
  if q in qrules:
   rem=tuple(m[z] for z in range(4) if z not in ij); item=('q',q,rem)
   if item not in ans:ans.append(item)
 return ans
def cdivs4(m):
 ans=[]
 for omit in range(4):
  c=tuple(m[z] for z in range(4) if z!=omit)
  if c in crules:
   item=('c',c,m[omit])
   if item not in ans:ans.append(item)
 return ans
def apps4(m):return qdivs4(m)+cdivs4(m)
standard4=[]
for m in itertools.combinations_with_replacement(range(150),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
assert len(standard4)==23817; s4index={m:i for i,m in enumerate(standard4)}
def parse_app(a):
 return (a[0],tuple(a[1]),tuple(a[2]) if isinstance(a[2],list) else int(a[2]))
raw=[(tuple(z[0]),parse_app(z[1])) for z in json.load(open('y36_quartic_raw_relations.json'))];assert len(raw)==87
def make_engine(Proj):
 dim=Proj.shape[1];cache={};visiting=set()
 def rewrite(m,a):
  typ,lead,rem=a;out=np.zeros(dim,dtype=np.uint64)
  if typ=='q':
   for z,(coef,b) in enumerate(qrules[lead]):
    cc=tuple(sorted((rem[0],rem[1],b[0],b[1])));out += np.uint64(coef)*nf(cc)
    if z%16==15:out%=p
  else:
   coefs,_=crules[lead]
   for z,ci in enumerate(cpivs):
    coef=int(coefs[z])
    if coef:
     c=standard3[ci];cc=tuple(sorted((rem,c[0],c[1],c[2])));out += np.uint64(coef)*nf(cc)
    if z%16==15:out%=p
  out%=p;return out
 def nf(m):
  i=s4index.get(m)
  if i is not None:return Proj[i]
  v=cache.get(m)
  if v is not None:return v
  if m in visiting:raise RuntimeError('cycle')
  aa=apps4(m);assert aa
  visiting.add(m);v=rewrite(m,aa[0]);visiting.remove(m);cache[m]=v;return v
 return nf,rewrite,cache
old=np.load('y36_quartic_profiles.npz')['profile_chunks']; assert old.shape==(87,15744)
# Independently verify that the durable raw-relation file has exactly the same
# row order as the earlier high-coordinate profile artifact.  We recompute the
# full invertible 87-by-87 coordinate minor at the saved leading coordinates;
# equality here certifies all row identities, not merely a sample.
leadcheck=list(map(int,np.load('y36_quartic_profiles.npz')['q4lead_indices']))
Check=np.zeros((23817,87),dtype=np.uint64)
for j,i in enumerate(leadcheck):Check[i,j]=1
cnf,crew,ccache=make_engine(Check);checkrows=[]
for m,alt in raw:
 v=(crew(m,alt).astype(np.int64)-cnf(m).astype(np.int64))%p;checkrows.append(v.astype(np.uint32))
oldcols=[23816-i for i in leadcheck]
assert np.array_equal(np.array(checkrows,dtype=np.uint32),old[:,oldcols])
del Check,cnf,crew,ccache,checkrows;gc.collect();print('verified raw/profile alignment on the invertible 87x87 lead-coordinate minor',flush=True)
# The old columns correspond in order to coordinate indices 23816 down to 8073.
assert 23817-old.shape[1]==8073
lower=[]; done=0
while done<8073 and (not maxchunks or len(lower)<maxchunks):
 indices=list(range(8072-done,max(-1,8072-done-chunk),-1)); assert indices
 Proj=np.zeros((23817,len(indices)),dtype=np.uint64)
 for j,i in enumerate(indices):Proj[i,j]=1
 nf,rewrite,cache=make_engine(Proj); rows=[]
 for z,(m,alt) in enumerate(raw):
  v=(rewrite(m,alt).astype(np.int64)-nf(m).astype(np.int64))%p; rows.append(v.astype(np.uint32))
 M=np.array(rows,dtype=np.uint32);lower.append(M);done+=len(indices)
 print(f'lower profiled={done}/8073 chunks={len(lower)} cache={len(cache)} nz={np.count_nonzero(M)} elapsed={time.time()-t:.1f}s',flush=True)
 del Proj,nf,rewrite,cache,M;gc.collect()
if done<8073:
 np.savez_compressed('y36_quartic_lower_partial.npz',lower_desc=np.concatenate(lower,axis=1),done=np.array([done]));sys.exit(0)
lower_desc=np.concatenate(lower,axis=1);assert lower_desc.shape==(87,8073)
all_desc=np.concatenate([old,lower_desc],axis=1); full=all_desc[:,::-1].copy();assert full.shape==(87,23817)
np.savez_compressed('y36_quartic_full_profiles.npz',relation_matrix=full)
h=hashlib.sha256(full.astype('<u4').tobytes()).hexdigest()
res={'prime':p,'shape':list(full.shape),'nonzeros':int(np.count_nonzero(full)),'profile_alignment_verified_on_lead_minor':True,'matrix_sha256':h,'artifact_sha256':hashlib.sha256(open('y36_quartic_full_profiles.npz','rb').read()).hexdigest(),'elapsed_seconds':time.time()-t}
open('y36_quartic_full_profiles_result.json','w').write(json.dumps(res,indent=2,sort_keys=True));print(json.dumps(res,indent=2,sort_keys=True),flush=True)
