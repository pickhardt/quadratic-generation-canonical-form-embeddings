#!/usr/bin/env python3
# Recover genuine leading monomials of the 87 quartic confluence relations
# and measure their degree-five monomial shadow.
import gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time()
rawtarget=int(os.getenv('RAWTARGET','87'));projdim=int(os.getenv('PROJDIM','534'));ranktarget=int(os.getenv('RANKTARGET',str(projdim)));maxx=int(os.getenv('MAXX','150'))
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
   if len(raw)==rawtarget:break
 if len(raw)==rawtarget:break
assert len(raw)==rawtarget
if rawtarget==87: assert tested==228748
# Free the 1.7m-vector cache before coordinate profiling.
del Proj,nf,rewrite,cache,ebasis;gc.collect()

# Save a portable description of the selected raw ambiguity identities.
def app_json(a):
 return [a[0],list(a[1]),list(a[2]) if isinstance(a[2],tuple) else a[2]]
raw_json=[[list(m),app_json(a)] for m,a in raw]
open('y36_quartic_raw_relations.json','w').write(json.dumps(raw_json,separators=(',',':')))
print(f'Building degree-five q/c-standard sector; projdim={projdim} target={ranktarget}.',flush=True)
# A quintic is standard modulo the degree-2 and degree-3 leading layers iff
# each of its quartic divisors belongs to standard4.
std4set=set(standard4);standard5=set()
for m in standard4:
 for x in range(150):
  mm=tuple(sorted(m+(x,)))
  if all(tuple(mm[j] for j in range(5) if j!=omit) in std4set for omit in range(5)):
   standard5.add(mm)
standard5=sorted(standard5);assert len(standard5)==56875
s5index={m:i for i,m in enumerate(standard5)}
ng5=np.random.default_rng(314159265)
Proj5=ng5.integers(0,p,size=(len(standard5),projdim),dtype=np.uint32)
projection_hash=hashlib.sha256(Proj5.astype('<u4').tobytes()).hexdigest()
cache5={};visiting5=set();calls5={'q':0,'c':0}

def firstapp5(m):
 for ij in itertools.combinations(range(5),2):
  q=(m[ij[0]],m[ij[1]])
  if q in qrules:
   rem=tuple(m[z] for z in range(5) if z not in ij);return ('q',q,rem)
 for tri in itertools.combinations(range(5),3):
  c=tuple(m[z] for z in tri)
  if c in crules:
   rem=tuple(m[z] for z in range(5) if z not in tri);return ('c',c,rem)
 return None

def rewrite5(a):
 typ,lead,rem=a;out=np.zeros(projdim,dtype=np.uint64);calls5[typ]+=1
 if typ=='q':
  for z,(coef,b) in enumerate(qrules[lead]):
   mm=tuple(sorted(rem+b));out += np.uint64(coef)*nf5(mm).astype(np.uint64)
   if z%64==63:out%=p
 else:
  coefs,_=crules[lead]
  for z,ci in enumerate(cpivs):
   coef=int(coefs[z])
   if coef:
    c=standard3[ci];mm=tuple(sorted(rem+c));out += np.uint64(coef)*nf5(mm).astype(np.uint64)
   if z%128==127:out%=p
 out%=p;return out.astype(np.uint32)

def nf5(m):
 i=s5index.get(m)
 if i is not None:return Proj5[i]
 v=cache5.get(m)
 if v is not None:return v
 if m in visiting5:raise RuntimeError('nontermination5 '+repr(m))
 a=firstapp5(m);assert a is not None
 visiting5.add(m);v=rewrite5(a);visiting5.remove(m);cache5[m]=v;return v

def app4_times_x(a,x):
 # Project the q/c-normal form of x times one one-step quartic rewrite.
 typ,lead,rem=a;out=np.zeros(projdim,dtype=np.uint64)
 if typ=='q':
  for z,(coef,b) in enumerate(qrules[lead]):
   mm=tuple(sorted(rem+b+(x,)));out += np.uint64(coef)*nf5(mm).astype(np.uint64)
   if z%64==63:out%=p
 else:
  coefs,_=crules[lead]
  for z,ci in enumerate(cpivs):
   coef=int(coefs[z])
   if coef:
    c=standard3[ci];mm=tuple(sorted((rem,x)+c));out += np.uint64(coef)*nf5(mm).astype(np.uint64)
   if z%128==127:out%=p
 out%=p;return out.astype(np.uint32)

# Images of x*r for the 87 independent quartic relations r.  Their rank in
# S_5/(degree <=3 relations) is at most 534 geometrically and is certified
# from below by this explicit projection.
eb5={};accepted5=[];tested5=0
for x in range(maxx):
 for ri,(m,alt) in enumerate(raw):
  base=apps4(m)[0]
  v=(app4_times_x(alt,x).astype(np.int64)-app4_times_x(base,x).astype(np.int64))%p
  v=v.astype(np.uint64);tested5+=1
  for k,b in list(eb5.items()):
   aa=int(v[k])
   if aa:v=(v+np.uint64(p-aa)*b)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;eb5[k]=v;accepted5.append((x,ri,k))
   print(f'quintic rank={len(eb5)} tested={tested5} x={x} cache={len(cache5)} qcalls={calls5["q"]} ccalls={calls5["c"]} elapsed={time.time()-t:.1f}s',flush=True)
   if len(eb5)>=ranktarget:break
 if len(eb5)>=ranktarget:break
 print(f'quintic heartbeat x={x} rank={len(eb5)} tested={tested5} cache={len(cache5)} elapsed={time.time()-t:.1f}s',flush=True)
ah=hashlib.sha256()
for x,ri,k in accepted5:ah.update(struct.pack('<III',x,ri,k))
bh=hashlib.sha256()
for k in sorted(eb5):bh.update(struct.pack('<I',k));bh.update(eb5[k].astype('<u4').tobytes())
res={'prime':p,'seed':seed,'raw_quartic_relations':len(raw),'degree_5_projection_dimension':projdim,'projection_sha256':projection_hash,'projected_quintic_rank':len(eb5),'rank_target':ranktarget,'quintic_generators_tested':tested5,'variables_processed':x+1,'q_c_standard_quintics':len(standard5),'certified_final_quotient_upper_bound':len(standard5)-len(eb5),'geometric_target':56341,'normal_forms_cached':len(cache5),'accepted_generators_sha256':ah.hexdigest(),'projected_echelon_basis_sha256':bh.hexdigest(),'elapsed_seconds':time.time()-t}
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
if rawtarget==87 and projdim==534 and ranktarget==534:
 open('y36_quintic_rank_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
if len(eb5)<ranktarget:sys.exit(2)
