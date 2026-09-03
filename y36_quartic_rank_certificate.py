#!/usr/bin/env python3
# Exact finite-field degree-4 certificate for the quadratic ideal of HPP Y(3,6).
# Uses quadratic rewrite rules plus the complete 706-dimensional cubic
# confluence space, then certifies 87 independent quartic confluence residuals
# after an explicit projection.
import argparse,gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
ap=argparse.ArgumentParser();ap.add_argument('--projdim',type=int,default=87);ap.add_argument('--target',type=int,default=87);ap.add_argument('--max-tested',type=int,default=1000000);args=ap.parse_args()
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time()
D=np.load('y36_cubic_basis_data.npz');coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
assert len(coord)==150 and len(pivs)==1720 and len(cpivs)==7876 and len(cnonp)==706
print(f'Expected: reconstruct quadrics (~25s), build rewrite system, seek {args.target} projected quartic residuals.',flush=True)
# Reconstruct exact quadratic RREF, checking the stored pivot profile.
coord2=y.pivot_columns(fmpz_mod_mat([y.random_point(rng,p) for _ in range(170)],ctx));assert coord2==coord
Y=[]
for z in range(1720):
 f=y.random_point(rng,p);Y.append([f[i] for i in coord])
logical=y.grevlex_degree2_increasing(150);mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical];col={m:i for i,m in enumerate(mons)}
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1720
# Verify/extract rules nonpivot monomial -> sparse list (coefficient,pivot monomial).
P=set(pivs);qrules={};qnnz=0
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 qrules[m]=rr;qnnz+=len(rr)
del R;gc.collect();assert len(qrules)==9605
# Cubic rules from the exact 7876x8582 evaluation RREF artifact.
crules={}
for z,j in enumerate(cnonp):crules[standard3[j]]=(ctails[z],z)
# Enumerate standard quartics under both leading sets.
def qdivs(m):
 ans=[]
 for ij in itertools.combinations(range(4),2):
  q=(m[ij[0]],m[ij[1]])
  if q in qrules:
   rem=tuple(m[z] for z in range(4) if z not in ij);item=('q',q,rem)
   if item not in ans:ans.append(item)
 return ans
def cdivs(m):
 ans=[]
 for omit in range(4):
  c=tuple(m[z] for z in range(4) if z!=omit)
  if c in crules:
   item=('c',c,m[omit])
   if item not in ans:ans.append(item)
 return ans
def apps(m):return qdivs(m)+cdivs(m)
standard4=[]
qstandard4=[]
for m in itertools.combinations_with_replacement(range(150),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 qstandard4.append(m)
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
assert len(qstandard4)==28569 and len(standard4)==23817
s4index={m:i for i,m in enumerate(standard4)}
print(f'rules q={len(qrules)} q_nnz={qnnz} c={len(crules)} standard4={len(standard4)} elapsed={time.time()-t:.1f}s',flush=True)
# Explicit deterministic projection from the standard quartic sector.
ng=np.random.default_rng(271828182);Proj=ng.integers(0,p,size=(len(standard4),args.projdim),dtype=np.uint64)
ph=hashlib.sha256(Proj.astype('<u4').tobytes()).hexdigest()
cache={};visiting=set();calls={'q':0,'c':0}
def nf(m):
 i=s4index.get(m)
 if i is not None:return Proj[i]
 v=cache.get(m)
 if v is not None:return v
 if m in visiting:raise RuntimeError('nonterminating rewrite at '+repr(m))
 aa=apps(m);assert aa
 visiting.add(m);v=rewrite(m,aa[0]);visiting.remove(m);cache[m]=v
 return v
def rewrite(m,a):
 typ,lead,rem=a;out=np.zeros(args.projdim,dtype=np.uint64);calls[typ]+=1
 if typ=='q':
  terms=qrules[lead]
  for z,(coef,b) in enumerate(terms):
   cc=tuple(sorted((rem[0],rem[1],b[0],b[1])))
   out += np.uint64(coef)*nf(cc)
   if z%16==15:out%=p
 else:
  coefs,_=crules[lead]
  for z,ci in enumerate(cpivs):
   coef=int(coefs[z])
   if coef:
    c=standard3[ci];cc=tuple(sorted((rem,c[0],c[1],c[2])))
    out += np.uint64(coef)*nf(cc)
   if z%16==15:out%=p
 out%=p;return out
# Online row echelon of projected ambiguity residuals.
ebasis={};tested=0;accepted=[]
# Critical quartics: first cubic/cubic overlaps among q-standard monomials,
# then quadratic/cubic overlaps generated as lcms of a cubic lead and a variable.
cc_candidates=[m for m in qstandard4 if len(cdivs(m))>1]
qc_set=set()
for c in crules:
 for x in range(150):
  m=tuple(sorted((x,c[0],c[1],c[2])))
  if qdivs(m) and cdivs(m):qc_set.add(m)
candidates=cc_candidates+sorted(qc_set-set(cc_candidates))
print(f'critical candidates cc={len(cc_candidates)} qc={len(qc_set)} total={len(candidates)}',flush=True)
for m in candidates:
 aa=apps(m)
 if len(aa)<2:continue
 base=nf(m)
 for alt in aa[1:]:
  v=(rewrite(m,alt).astype(np.int64)-base.astype(np.int64))%p;v=v.astype(np.uint64);tested+=1
  for k,b in list(ebasis.items()):
   a=int(v[k])
   if a:v=(v+np.uint64(p-a)*b)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;ebasis[k]=v;accepted.append((m,alt,k))
   print(f'rank={len(ebasis)} tested={tested} cache={len(cache)} qcalls={calls["q"]} ccalls={calls["c"]} elapsed={time.time()-t:.1f}s',flush=True)
   if len(ebasis)>=args.target:break
  if tested>=args.max_tested:break
 if len(ebasis)>=args.target or tested>=args.max_tested:break
 if tested and tested%1000==0:print(f'tested={tested} rank={len(ebasis)} cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
ah=hashlib.sha256()
for m,a,k in accepted:
 ah.update(bytes(m));ah.update(a[0].encode());ah.update(bytes(a[1] if isinstance(a[1],tuple) else (a[1],)));r=a[2] if isinstance(a[2],tuple) else (a[2],);ah.update(bytes(r));ah.update(struct.pack('<I',k))
bh=hashlib.sha256()
for k in sorted(ebasis):bh.update(struct.pack('<I',k));bh.update(ebasis[k].astype('<u4').tobytes())
res={'prime':p,'seed':seed,'projection_dimension':args.projdim,'projection_sha256':ph,'quadratic_rules':len(qrules),'cubic_rules':len(crules),'standard_quartics_after_degree_2_3_leads':len(standard4),'quartic_confluence_rank':len(ebasis),'certified_quadratic_quotient_upper_bound':len(standard4)-len(ebasis),'residuals_tested':tested,'normal_forms_cached':len(cache),'accepted_residuals_sha256':ah.hexdigest(),'projected_echelon_basis_sha256':bh.hexdigest(),'elapsed_seconds':time.time()-t}
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
if args.projdim==87 and args.target==87:
 open('y36_quartic_rank_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
if len(ebasis)<args.target:sys.exit(2)
