#!/usr/bin/env python3
# Exact final quintic certificate for Y(3,6): orient the 87 genuine quartic
# leads, count their 529-monomial shadow, and find five independent degree-5
# confluence residuals after reduction by the degree 2/3/4 rules.
import gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time()
ranktarget=int(os.getenv('RANKTARGET','5'));maxcand=int(os.getenv('MAXCAND','0'))
D=np.load('y36_cubic_basis_data.npz');coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
print(f'Expected: reconstruction <1 min, then search sparse q/c/q4 critical quintics for rank {ranktarget}.',flush=True)
# Reconstruct the exact quadratic rules.
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
# Sparse cubic rules.
crules={}
for z,j in enumerate(cnonp):
 nz=np.flatnonzero(ctails[z]);crules[standard3[j]]=[(int(ctails[z,i]),standard3[cpivs[i]]) for i in nz]
# The natural combinations_with_replacement enumeration is increasing grevlex
# after the recorded reversal perm[i]=149-i.  This follows directly by mapping
# exponent vectors under variable reversal; we also test degrees 2--5 below.
def grevlex_cmp_actual(a,b,n=150):
 # +1 iff a is larger in the fixed grevlex order whose variables are
 # x_149 > ... > x_0 (the actual labels after reversal).
 ea=[0]*n;eb=[0]*n
 for i in a:ea[i]+=1
 for i in b:eb[i]+=1
 for i in range(n): # last logical variable is actual 0, then 1, ...
  if ea[i]!=eb[i]:return 1 if ea[i]<eb[i] else -1
 return 0
for d in range(2,6):
 test=list(itertools.combinations_with_replacement(range(7),d))
 assert all(grevlex_cmp_actual(test[i],test[i+1],7)<0 for i in range(len(test)-1))
# Standard quartics under lower leads.
standard4=[]
for m in itertools.combinations_with_replacement(range(150),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
assert len(standard4)==23817;s4index={m:i for i,m in enumerate(standard4)};std4set=set(standard4)
O=np.load('y36_quartic_oriented_rules.npz');q4lead_indices=list(map(int,O['leads']));G=O['rules'];q4leads=[standard4[i] for i in q4lead_indices];q4leadset=set(q4leads)
# Verify exact orientation, genuine leading property, and make sparse rewrite tails.
assert np.array_equal(G[:,q4lead_indices],np.eye(87,dtype=np.uint32))
q4rules={}
for i,lead in enumerate(q4leads):
 nz=np.flatnonzero(G[i]);assert int(nz[-1])==q4lead_indices[i]
 tail=[]
 for j in nz:
  if int(j)==q4lead_indices[i]:continue
  assert int(j)<q4lead_indices[i]
  c=(-int(G[i,j]))%p
  if c:tail.append((c,standard4[int(j)]))
 q4rules[lead]=tail
# Build the 56,875 lower-standard quintics and the 56,346 final monomial complement.
standard5=set()
for m in standard4:
 for x in range(150):
  mm=tuple(sorted(m+(x,)))
  if all(tuple(mm[j] for j in range(5) if j!=omit) in std4set for omit in range(5)):standard5.add(mm)
assert len(standard5)==56875
final5=sorted(m for m in standard5 if not any(tuple(m[j] for j in range(5) if j!=omit) in q4leadset for omit in range(5)))
assert len(final5)==56346;f5index={m:i for i,m in enumerate(final5)}
print(f'rules ready: q={len(qrules)} c={len(crules)} q4={len(q4rules)}; quintic shadow={len(standard5)-len(final5)}; elapsed={time.time()-t:.1f}s',flush=True)
# Fixed exact projection of the final monomial complement to F_p^5.
ng=np.random.default_rng(161803398);Proj=ng.integers(0,p,size=(len(final5),5),dtype=np.uint32)
projection_hash=hashlib.sha256(Proj.astype('<u4').tobytes()).hexdigest()
cache={};visiting=set();calls={'q':0,'c':0,'q4':0}
def actions5(m):
 ans=[]
 for ij in itertools.combinations(range(5),2):
  lead=(m[ij[0]],m[ij[1]])
  if lead in qrules:
   rem=tuple(m[z] for z in range(5) if z not in ij);a=('q',lead,rem)
   if a not in ans:ans.append(a)
 for tri in itertools.combinations(range(5),3):
  lead=tuple(m[z] for z in tri)
  if lead in crules:
   rem=tuple(m[z] for z in range(5) if z not in tri);a=('c',lead,rem)
   if a not in ans:ans.append(a)
 for omit in range(5):
  lead=tuple(m[z] for z in range(5) if z!=omit)
  if lead in q4rules:
   a=('q4',lead,m[omit])
   if a not in ans:ans.append(a)
 return ans
def rewrite5(a):
 typ,lead,rem=a;calls[typ]+=1;out=np.zeros(5,dtype=np.uint64)
 if typ=='q':
  terms=qrules[lead]
  for z,(coef,b) in enumerate(terms):
   out+=np.uint64(coef)*nf5(tuple(sorted(rem+b)))
   if z%64==63:out%=p
 elif typ=='c':
  terms=crules[lead]
  for z,(coef,c) in enumerate(terms):
   out+=np.uint64(coef)*nf5(tuple(sorted(rem+c)))
   if z%64==63:out%=p
 else:
  terms=q4rules[lead]
  for z,(coef,q4) in enumerate(terms):
   out+=np.uint64(coef)*nf5(tuple(sorted(q4+(rem,))))
   if z%64==63:out%=p
 out%=p;return out.astype(np.uint32)
def nf5(m):
 i=f5index.get(m)
 if i is not None:return Proj[i]
 v=cache.get(m)
 if v is not None:return v
 if m in visiting:raise RuntimeError('nontermination '+repr(m))
 aa=actions5(m);assert aa
 visiting.add(m);v=rewrite5(aa[0]);visiting.remove(m);cache[m]=v;return v
# All possible new critical ambiguities occur among variable multiples of the
# quartic leads.  Compare every alternate one-step reduction with the chosen
# reduction and echelonize the fully reduced residuals.
candidates=sorted({tuple(sorted(lead+(x,))) for lead in q4leads for x in range(150)})
eb={};accepted=[];tested=0
for ci,m in enumerate(candidates):
 aa=actions5(m);assert any(a[0]=='q4' for a in aa)
 if len(aa)<2:continue
 base=nf5(m)
 for ai,alt in enumerate(aa[1:],1):
  v=(rewrite5(alt).astype(np.int64)-base.astype(np.int64))%p;v=v.astype(np.uint64);tested+=1
  for k,b in list(eb.items()):
   a=int(v[k])
   if a:v=(v+np.uint64(p-a)*b)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;eb[k]=v.astype(np.uint32);accepted.append((m,ai,k))
   print(f'residual rank={len(eb)} tested={tested} candidate={ci+1}/{len(candidates)} cache={len(cache)} calls={calls} elapsed={time.time()-t:.1f}s',flush=True)
   if len(eb)>=ranktarget:break
 if len(eb)>=ranktarget:break
 if maxcand and ci+1>=maxcand:break
 if (ci+1)%100==0:print(f'heartbeat candidate={ci+1}/{len(candidates)} rank={len(eb)} tested={tested} cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
# Save the tiny residual certificate itself (critical monomial, alternate-action
# index, pivot coordinate, and the 5-dimensional projected echelon rows).
accepted_json=[{'monomial':list(m),'alternate_action_index':ai,'pivot_coordinate':k} for m,ai,k in accepted]
open('y36_quintic_residual5_accepted.json','w').write(json.dumps(accepted_json,indent=2,sort_keys=True))
ekeys=sorted(eb);emat=np.array([eb[k] for k in ekeys],dtype=np.uint32)
np.savez_compressed('y36_quintic_residual5_echelon.npz',pivot_coordinates=np.array(ekeys,dtype=np.int32),echelon_rows=emat)
# Hash certificate data.
ah=hashlib.sha256()
for m,ai,k in accepted:ah.update(bytes(m));ah.update(struct.pack('<II',ai,k))
bh=hashlib.sha256()
for k in sorted(eb):bh.update(struct.pack('<I',k));bh.update(eb[k].astype('<u4').tobytes())
res={'prime':p,'seed':seed,'quartic_rules':87,'quartic_leads_sha256':hashlib.sha256(b''.join(bytes(m) for m in sorted(q4leadset))).hexdigest(),'standard_quintics_before_quartic_leads':len(standard5),'quartic_monomial_shadow':len(standard5)-len(final5),'standard_quintics_after_quartic_leads':len(final5),'geometric_target':56341,'residual_rank':len(eb),'rank_target':ranktarget,'certified_quotient_upper_bound':len(final5)-len(eb),'critical_residuals_tested':tested,'critical_candidates_reached':ci+1,'normal_forms_cached':len(cache),'projection_sha256':projection_hash,'accepted_residuals_sha256':ah.hexdigest(),'projected_echelon_sha256':bh.hexdigest(),'accepted_artifact_sha256':hashlib.sha256(open('y36_quintic_residual5_accepted.json','rb').read()).hexdigest(),'echelon_artifact_sha256':hashlib.sha256(open('y36_quintic_residual5_echelon.npz','rb').read()).hexdigest(),'calls':calls,'elapsed_seconds':time.time()-t}
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
if ranktarget==5 and len(eb)==5:
 open('y36_quintic_residual5_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
if len(eb)<ranktarget:sys.exit(2)
