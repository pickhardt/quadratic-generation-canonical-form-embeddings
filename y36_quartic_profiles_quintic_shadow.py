#!/usr/bin/env python3
# Recover genuine leading monomials of the 87 quartic confluence relations
# and measure their degree-five monomial shadow.
import gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time()
D=np.load('y36_cubic_basis_data.npz');coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
assert len(coord)==150 and len(pivs)==1720 and len(cpivs)==7876 and len(cnonp)==706
print('Expected: ~4 min recover raw quartic ambiguities; then inspect exact highest-coordinate profiles.',flush=True)
# Reconstruct quadratic rules exactly.
coord2=y.pivot_columns(fmpz_mod_mat([y.random_point(rng,p) for _ in range(170)],ctx));assert coord2==coord
Y=[]
for z in range(1720):
 f=y.random_point(rng,p);Y.append([f[i] for i in coord])
logical=y.grevlex_degree2_increasing(150);mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1720
P=set(pivs);qrules={}
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 qrules[m]=rr
del R;gc.collect();assert len(qrules)==9605
crules={}
for z,j in enumerate(cnonp):crules[standard3[j]]=(ctails[z],z)

def qdivs4(m):
 ans=[]
 for ij in itertools.combinations(range(4),2):
  q=(m[ij[0]],m[ij[1]])
  if q in qrules:
   rem=tuple(m[z] for z in range(4) if z not in ij);item=('q',q,rem)
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
standard4=[];qstandard4=[]
for m in itertools.combinations_with_replacement(range(150),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 qstandard4.append(m)
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
assert len(qstandard4)==28569 and len(standard4)==23817
s4index={m:i for i,m in enumerate(standard4)}
cc_candidates=[m for m in qstandard4 if len(cdivs4(m))>1]
qc_set=set()
for c in crules:
 for x in range(150):
  m=tuple(sorted((x,c[0],c[1],c[2])))
  if qdivs4(m) and cdivs4(m):qc_set.add(m)
candidates=cc_candidates+sorted(qc_set-set(cc_candidates))
print(f'rules reconstructed; standard4={len(standard4)} candidates={len(candidates)} elapsed={time.time()-t:.1f}s',flush=True)

# Generic projected normal-form engine for quartics.
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
  if m in visiting:raise RuntimeError('nontermination '+repr(m))
  aa=apps4(m);assert aa
  visiting.add(m);v=rewrite(m,aa[0]);visiting.remove(m);cache[m]=v;return v
 return nf,rewrite,cache

# Phase A: exactly reproduce the original random-projection choice of 87
# linearly independent *raw* ambiguity residuals.
ng=np.random.default_rng(271828182);Proj=ng.integers(0,p,size=(len(standard4),87),dtype=np.uint64)
nf,rewrite,cache=make_engine(Proj);ebasis={};raw=[];tested=0
for m in candidates:
 aa=apps4(m)
 if len(aa)<2:continue
 base=nf(m)
 for alt in aa[1:]:
  v=(rewrite(m,alt).astype(np.int64)-base.astype(np.int64))%p;v=v.astype(np.uint64);tested+=1
  for k,b in list(ebasis.items()):
   a=int(v[k])
   if a:v=(v+np.uint64(p-a)*b)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;ebasis[k]=v;raw.append((m,alt))
   print(f'phaseA rank={len(raw)} tested={tested} cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
   if len(raw)==87:break
 if len(raw)==87:break
assert len(raw)==87 and tested==228748
# Free the 1.7m-vector cache before coordinate profiling.
del Proj,nf,rewrite,cache,ebasis;gc.collect()

# Phase B: inspect standard-quartic coordinates from highest to lowest in
# bounded chunks.  This both controls memory and makes accepted coordinate
# pivots genuine leading monomials for the fixed order.
chunk=128; colbasis={}; q4lead_indices=[]; scanned=0; profile_chunks=[]
while len(colbasis)<87:
 indices=list(range(len(standard4)-1-scanned,max(-1,len(standard4)-1-scanned-chunk),-1))
 assert indices
 Proj=np.zeros((len(standard4),len(indices)),dtype=np.uint64)
 for j,i in enumerate(indices):Proj[i,j]=1
 nf,rewrite,cache=make_engine(Proj);M=[]
 for z,(m,alt) in enumerate(raw):
  v=(rewrite(m,alt).astype(np.int64)-nf(m).astype(np.int64))%p
  M.append(v.astype(np.uint64))
  if (z+1)%20==0:print(f'phaseB chunkstart={scanned} residuals={z+1}/87 cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
 M=np.array(M,dtype=np.uint64)
 # Columns are coordinate functionals on the 87-dimensional raw relation space.
 # Scan in decreasing monomial order and retain independent columns.
 for j,i in enumerate(indices):
  v=M[:,j].copy()
  for k,bv in colbasis.items():
   aa=int(v[k])
   if aa:v=(v+np.uint64(p-aa)*bv)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;colbasis[k]=v;q4lead_indices.append(i)
   if len(colbasis)==87:break
 profile_chunks.append(M.astype(np.uint32));scanned+=len(indices)
 print(f'phaseB scanned={scanned} coordinate-rank={len(colbasis)} cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
 del Proj,nf,rewrite,cache,M;gc.collect()
assert len(colbasis)==87
rank=87;q4leads={standard4[i] for i in q4lead_indices}
# Degree-five standard monomials under degree 2/3 leads are generated by
# adjoining one variable to standard quartics and retaining those whose every
# quartic divisor is standard.
std4set=set(standard4);standard5=set()
for z,m in enumerate(standard4):
 for x in range(150):
  mm=tuple(sorted(m+(x,)))
  # all quartic divisors must avoid lower leading monomials
  if all(tuple(mm[j] for j in range(5) if j!=omit) in std4set for omit in range(5)):
   standard5.add(mm)
 if z and z%5000==0:print(f'quintic generation {z}/{len(standard4)} current={len(standard5)}',flush=True)
assert len(standard5)==56875
standard5q4=[m for m in standard5 if not any(tuple(m[j] for j in range(5) if j!=omit) in q4leads for omit in range(5))]
remaining=len(standard5q4);removed=len(standard5)-remaining
rawhash=hashlib.sha256();
for m,a in raw:
 rawhash.update(bytes(m));rawhash.update(a[0].encode());rawhash.update(bytes(a[1] if isinstance(a[1],tuple) else (a[1],)));rr=a[2] if isinstance(a[2],tuple) else (a[2],);rawhash.update(bytes(rr))
leadhash=hashlib.sha256(b''.join(bytes(m) for m in sorted(q4leads))).hexdigest()
res={'prime':p,'raw_quartic_relations':87,'raw_relations_sha256':rawhash.hexdigest(),'profile_coordinates_scanned':scanned,'profile_chunk_size':chunk,'profile_rank':rank,'quartic_leading_monomials':87,'quartic_leads_sha256':leadhash,'standard_quintics_before_quartic_leads':len(standard5),'quintics_removed_by_quartic_leads':removed,'standard_quintics_after_degree_2_3_4_leads':remaining,'target_hilbert_value':56341,'residual_gap':remaining-56341,'elapsed_seconds':time.time()-t}
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
open('y36_quartic_profiles_quintic_shadow_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
np.savez_compressed('y36_quartic_profiles.npz',q4lead_indices=np.array(q4lead_indices,dtype=np.int32),profile_chunks=np.concatenate(profile_chunks,axis=1))
