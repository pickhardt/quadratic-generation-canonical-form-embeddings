#!/usr/bin/env python3
# Exact final degree-5 certificate for X_cf(3,6): orient 81 quartic leads,
# count their monomial shadow, and find all remaining confluence directions
# required to force the quadratic quotient to geometric dimension h^0(5L)=44196.
import gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time();GEOM=44196
D=np.load('x36_cubic_basis_data.npz');coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
# Compile HPP X functions exactly as in the degree 2--4 certificates.
tails=[];seen=set()
for q in itertools.permutations(range(1,6)):
 if q in seen:continue
 tails.append(q);seen.add(q);seen.add(tuple(reversed(q)))
SIGMA=[(0,)+q for q in tails]
def compile_x(s):
 def P(tt):return y.signed_index(tuple(s[i-1] for i in tt))
 B=tuple(P(tt) for tt in [(1,2,3),(1,2,6),(1,4,5),(2,3,4),(3,5,6),(4,5,6)])
 C=(P((2,4,5)),tuple(P(tt) for tt in [(1,2,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,5,6),(4,5,6)]))
 q1=tuple(P(tt) for tt in [(1,2,3),(3,4,5),(1,5,6),(2,4,6)]);q2=tuple(P(tt) for tt in [(2,3,4),(4,5,6),(1,2,6),(1,3,5)])
 DD=(q1,q2,tuple(P(tt) for tt in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
 cyc=tuple(y.signed_index((s[i],s[(i+1)%6],s[(i+2)%6])) for i in range(6));return cyc,B,C,DD
COMP=[compile_x(s) for s in SIGMA]
def all_forms(X):
 pl=[y.det3cols(X,z)%p for z in y.TRIPLES];Ds=[];Cs=[];Bs=[];As=[]
 for A,B,C,DD in COMP:
  Ds.append(((y.prodP(pl,DD[0],p)-y.prodP(pl,DD[1],p))%p,y.prodP(pl,DD[2],p)))
  Cs.append((y.pv(pl,C[0],p),y.prodP(pl,C[1],p)));Bs.append((1,y.prodP(pl,B,p)));As.append((1,y.prodP(pl,A,p)))
 pairs=Ds+Cs+Bs+As;inv=y.batch_invert([b for a,b in pairs],p);return [(a*inv[i])%p for i,(a,b) in enumerate(pairs)]
def random_point():
 while True:
  X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
  try:return all_forms(X)
  except ZeroDivisionError:pass
print('Expected: reconstruction <1 min, quintic enumeration, then projected exact confluence search.',flush=True)
coord2=y.pivot_columns(fmpz_mod_mat([random_point() for _ in range(140)],ctx));assert coord2==coord
Y=[]
for z in range(1385):
 f=random_point();Y.append([f[i] for i in coord])
logical=y.grevlex_degree2_increasing(126);mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1385
P=set(pivs);qrules={}
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 qrules[m]=rr
del R;gc.collect();assert len(qrules)==6616
crules={}
for z,j in enumerate(cnonp):
 nz=np.flatnonzero(ctails[z]);crules[standard3[j]]=[(int(ctails[z,i]),standard3[cpivs[i]]) for i in nz]
def grevlex_cmp_actual(a,b,n=126):
 ea=[0]*n;eb=[0]*n
 for i in a:ea[i]+=1
 for i in b:eb[i]+=1
 for i in range(n):
  if ea[i]!=eb[i]:return 1 if ea[i]<eb[i] else -1
 return 0
for d in range(2,6):
 test=list(itertools.combinations_with_replacement(range(7),d));assert all(grevlex_cmp_actual(test[i],test[i+1],7)<0 for i in range(len(test)-1))
standard4=[]
for m in itertools.combinations_with_replacement(range(126),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
assert len(standard4)==18776;std4set=set(standard4)
O=np.load('x36_quartic_oriented_rules.npz');q4lead_indices=list(map(int,O['leads']));G=O['rules'];q4leads=[standard4[i] for i in q4lead_indices];q4leadset=set(q4leads)
assert np.array_equal(G[:,q4lead_indices],np.eye(81,dtype=np.uint32))
q4rules={}
for i,lead in enumerate(q4leads):
 nz=np.flatnonzero(G[i]);assert int(nz[-1])==q4lead_indices[i]
 tail=[]
 for j in nz:
  if int(j)==q4lead_indices[i]:continue
  assert int(j)<q4lead_indices[i];c=(-int(G[i,j]))%p
  if c:tail.append((c,standard4[int(j)]))
 q4rules[lead]=tail
# Enumerate standard quintics after the degree 2/3 layer and after quartic leads.
standard5=set()
for ii,m in enumerate(standard4):
 for x in range(126):
  mm=tuple(sorted(m+(x,)))
  if all(tuple(mm[j] for j in range(5) if j!=omit) in std4set for omit in range(5)):standard5.add(mm)
 if (ii+1)%4000==0:print(f'quintic enumeration {ii+1}/{len(standard4)} elapsed={time.time()-t:.1f}s',flush=True)
final5=sorted(m for m in standard5 if not any(tuple(m[j] for j in range(5) if j!=omit) in q4leadset for omit in range(5)))
required_rank=len(final5)-GEOM
ranktarget=int(os.getenv('RANKTARGET',str(required_rank)))
print(f'rules q={len(qrules)} c={len(crules)} q4={len(q4rules)} standard5={len(standard5)} shadow={len(standard5)-len(final5)} final5={len(final5)} geometric={GEOM} ranktarget={ranktarget} elapsed={time.time()-t:.1f}s',flush=True)
assert required_rank>=0 and 0<=ranktarget<=required_rank
if os.getenv('COUNT_ONLY'):
 sys.exit(0)
f5index={m:i for i,m in enumerate(final5)}
# Fixed projection to precisely the dimension that must be killed.
ng=np.random.default_rng(161803399);Proj=ng.integers(0,p,size=(len(final5),ranktarget),dtype=np.uint32)
projection_hash=hashlib.sha256(Proj.astype('<u4').tobytes()).hexdigest();cache={};visiting=set();calls={'q':0,'c':0,'q4':0}
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
def nf5(m):
 i=f5index.get(m)
 if i is not None:return Proj[i]
 v=cache.get(m)
 if v is not None:return v
 if m in visiting:raise RuntimeError('nontermination '+repr(m))
 aa=actions5(m);assert aa
 visiting.add(m);v=rewrite5(aa[0]);visiting.remove(m);cache[m]=v;return v
def rewrite5(a):
 typ,lead,rem=a;calls[typ]+=1;out=np.zeros(ranktarget,dtype=np.uint64)
 if typ=='q':terms=qrules[lead]; join=lambda b:tuple(sorted(rem+b))
 elif typ=='c':terms=crules[lead]; join=lambda b:tuple(sorted(rem+b))
 else:terms=q4rules[lead];join=lambda b:tuple(sorted(b+(rem,)))
 for z,(coef,b) in enumerate(terms):
  out+=np.uint64(coef)*nf5(join(b))
  if z%32==31:out%=p
 out%=p;return out.astype(np.uint32)
candidates=sorted({tuple(sorted(lead+(x,))) for lead in q4leads for x in range(126)})
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
   print(f'residual rank={len(eb)}/{ranktarget} tested={tested} candidate={ci+1}/{len(candidates)} cache={len(cache)} calls={calls} elapsed={time.time()-t:.1f}s',flush=True)
   if len(eb)>=ranktarget:break
 if len(eb)>=ranktarget:break
 if (ci+1)%100==0:print(f'heartbeat candidate={ci+1}/{len(candidates)} rank={len(eb)}/{ranktarget} tested={tested} cache={len(cache)} elapsed={time.time()-t:.1f}s',flush=True)
accepted_json=[{'monomial':list(m),'alternate_action_index':ai,'pivot_coordinate':k} for m,ai,k in accepted]
open('x36_quintic_residual_accepted.json','w').write(json.dumps(accepted_json,indent=2,sort_keys=True))
ekeys=sorted(eb);emat=np.array([eb[k] for k in ekeys],dtype=np.uint32);np.savez_compressed('x36_quintic_residual_echelon.npz',pivot_coordinates=np.array(ekeys,dtype=np.int32),echelon_rows=emat)
ah=hashlib.sha256()
for m,ai,k in accepted:ah.update(bytes(m));ah.update(struct.pack('<II',ai,k))
bh=hashlib.sha256()
for k in sorted(eb):bh.update(struct.pack('<I',k));bh.update(eb[k].astype('<u4').tobytes())
res={'prime':p,'seed':seed,'quartic_rules':81,'quartic_leads_sha256':hashlib.sha256(b''.join(bytes(m) for m in sorted(q4leadset))).hexdigest(),'standard_quintics_before_quartic_leads':len(standard5),'quartic_monomial_shadow':len(standard5)-len(final5),'standard_quintics_after_quartic_leads':len(final5),'geometric_target':GEOM,'residual_rank':len(eb),'rank_target':ranktarget,'required_rank_for_geometric_target':required_rank,'certified_quotient_upper_bound':len(final5)-len(eb),'critical_residuals_tested':tested,'critical_candidates_reached':ci+1,'normal_forms_cached':len(cache),'projection_sha256':projection_hash,'accepted_residuals_sha256':ah.hexdigest(),'projected_echelon_sha256':bh.hexdigest(),'accepted_artifact_sha256':hashlib.sha256(open('x36_quintic_residual_accepted.json','rb').read()).hexdigest(),'echelon_artifact_sha256':hashlib.sha256(open('x36_quintic_residual_echelon.npz','rb').read()).hexdigest(),'calls':calls,'elapsed_seconds':time.time()-t}
print(json.dumps(res,indent=2,sort_keys=True),flush=True);open('x36_quintic_residual_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
if len(eb)<ranktarget:sys.exit(2)
