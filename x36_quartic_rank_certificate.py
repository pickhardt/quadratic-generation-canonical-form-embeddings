#!/usr/bin/env python3
# Exact degree-4 certificate for the quadratic ideal of HPP X(3,6).
# Uses the complete 780-dimensional cubic rewrite layer, then seeks exactly
# standard4-18695 independent quartic confluence residuals after projection.
import gc,hashlib,itertools,json,os,random,struct,sys,time
import numpy as np
from flint import fmpz_mod_ctx,fmpz_mod_mat
sys.path.insert(0,os.getcwd());import y36_cubic_sieve as y
p=1000003;seed=260728368;ctx=fmpz_mod_ctx(p);rng=random.Random(seed);t=time.time();TARGET_H4=18695
D=np.load('x36_cubic_basis_data.npz');coord=list(map(int,D['coord']));perm=list(map(int,D['perm']));pivs=list(map(int,D['quadratic_pivs']));standard3=[tuple(map(int,x)) for x in D['standard_cubics']];cpivs=list(map(int,D['cubic_pivs']));cnonp=list(map(int,D['cubic_nonp']));ctails=D['cubic_tails']
assert len(coord)==126 and len(pivs)==1385 and len(cpivs)==6250 and len(cnonp)==780
# Compile X canonical functions.
tails=[];seen=set()
for q in itertools.permutations(range(1,6)):
 if q in seen:continue
 tails.append(q);seen.add(q);seen.add(tuple(reversed(q)))
SIGMA=[(0,)+q for q in tails]
def compile_x(s):
 def P(t):return y.signed_index(tuple(s[i-1] for i in t))
 B=tuple(P(t) for t in [(1,2,3),(1,2,6),(1,4,5),(2,3,4),(3,5,6),(4,5,6)])
 C=(P((2,4,5)),tuple(P(t) for t in [(1,2,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,5,6),(4,5,6)]))
 q1=tuple(P(t) for t in [(1,2,3),(3,4,5),(1,5,6),(2,4,6)])
 q2=tuple(P(t) for t in [(2,3,4),(4,5,6),(1,2,6),(1,3,5)])
 DD=(q1,q2,tuple(P(t) for t in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
 cyc=tuple(y.signed_index((s[i],s[(i+1)%6],s[(i+2)%6])) for i in range(6))
 return cyc,B,C,DD
COMP=[compile_x(s) for s in SIGMA]
def all_forms(X):
 pl=[y.det3cols(X,z)%p for z in y.TRIPLES];Ds=[];Cs=[];Bs=[];As=[]
 for A,B,C,DD in COMP:
  Ds.append(((y.prodP(pl,DD[0],p)-y.prodP(pl,DD[1],p))%p,y.prodP(pl,DD[2],p)))
  Cs.append((y.pv(pl,C[0],p),y.prodP(pl,C[1],p)));Bs.append((1,y.prodP(pl,B,p)));As.append((1,y.prodP(pl,A,p)))
 pairs=Ds+Cs+Bs+As;inv=y.batch_invert([b for a,b in pairs],p)
 return [(a*inv[i])%p for i,(a,b) in enumerate(pairs)]
def random_point():
 while True:
  X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
  try:return all_forms(X)
  except ZeroDivisionError:pass
if os.getenv('SMOKE'):
 assert len(random_point())==240 and D['cubic_tails'].shape==(780,6250);print('SMOKE_OK',flush=True);sys.exit(0)
print('Expected: reconstruct quadrics (~10s), enumerate quartics, then projected confluence search.',flush=True)
coord2=y.pivot_columns(fmpz_mod_mat([random_point() for _ in range(140)],ctx));assert coord2==coord
Y=[]
for z in range(1385):
 f=random_point();Y.append([f[i] for i in coord])
logical=y.grevlex_degree2_increasing(126);mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
Q=fmpz_mod_mat([[(r[i]*r[j])%p for i,j in mons] for r in Y],ctx);del Y
R,rank=Q.rref();del Q;assert rank==1385
P=set(pivs);qrules={};qnnz=0
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 qrules[m]=rr;qnnz+=len(rr)
del R;gc.collect();assert len(qrules)==6616
crules={}
for z,j in enumerate(cnonp):crules[standard3[j]]=(ctails[z],z)
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
standard4=[];qstandard4=[]
for m in itertools.combinations_with_replacement(range(126),4):
 if any((m[i],m[j]) in qrules for i in range(4) for j in range(i+1,4)):continue
 qstandard4.append(m)
 if any(tuple(m[z] for z in range(4) if z!=omit) in crules for omit in range(4)):continue
 standard4.append(m)
needed=len(standard4)-TARGET_H4
print(f'rules q={len(qrules)} q_nnz={qnnz} c={len(crules)} qstandard4={len(qstandard4)} standard4={len(standard4)} target={TARGET_H4} needed={needed} elapsed={time.time()-t:.1f}s',flush=True)
assert needed>=0
if needed==0:sys.exit(0)
s4index={m:i for i,m in enumerate(standard4)}
ng=np.random.default_rng(271828182);Proj=ng.integers(0,p,size=(len(standard4),needed),dtype=np.uint64)
ph=hashlib.sha256(Proj.astype('<u4').tobytes()).hexdigest();cache={};visiting=set();calls={'q':0,'c':0}
def nf(m):
 i=s4index.get(m)
 if i is not None:return Proj[i]
 v=cache.get(m)
 if v is not None:return v
 if m in visiting:raise RuntimeError('nonterminating '+repr(m))
 aa=apps(m);assert aa
 visiting.add(m);v=rewrite(m,aa[0]);visiting.remove(m);cache[m]=v;return v
def rewrite(m,a):
 typ,lead,rem=a;out=np.zeros(needed,dtype=np.uint64);calls[typ]+=1
 if typ=='q':
  for z,(coef,b) in enumerate(qrules[lead]):
   cc=tuple(sorted((rem[0],rem[1],b[0],b[1])));out+=np.uint64(coef)*nf(cc)
   if z%16==15:out%=p
 else:
  coefs,_=crules[lead]
  for z,ci in enumerate(cpivs):
   coef=int(coefs[z])
   if coef:
    c=standard3[ci];cc=tuple(sorted((rem,c[0],c[1],c[2])));out+=np.uint64(coef)*nf(cc)
   if z%16==15:out%=p
 out%=p;return out
ebasis={};tested=0;accepted=[]
cc_candidates=[m for m in qstandard4 if len(cdivs(m))>1]
qc_set=set()
for c in crules:
 for x in range(126):
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
   if len(ebasis)%10==0 or len(ebasis)==needed:print(f'rank={len(ebasis)}/{needed} tested={tested} cache={len(cache)} qcalls={calls["q"]} ccalls={calls["c"]} elapsed={time.time()-t:.1f}s',flush=True)
   if len(ebasis)>=needed:break
 if len(ebasis)>=needed:break
 if tested and tested%10000==0:print(f'tested={tested} rank={len(ebasis)} elapsed={time.time()-t:.1f}s',flush=True)
ah=hashlib.sha256(repr(accepted).encode()).hexdigest();bh=hashlib.sha256()
for k in sorted(ebasis):bh.update(struct.pack('<I',k));bh.update(ebasis[k].astype('<u4').tobytes())
json.dump([[list(m),[a[0],list(a[1]),list(a[2]) if isinstance(a[2],tuple) else a[2]]] for m,a,k in accepted],open('x36_quartic_raw_relations.json','w'),indent=2)
if os.getenv('FULL_PROFILE'):
 required=needed; chunk=int(os.getenv('CHUNK','256')); maxchunks=int(os.getenv('MAXCHUNKS','0')); os.makedirs('x36_profile_chunks',exist_ok=True)
 blocks=[];made=0
 print(f'FULL_PROFILE begins: {len(accepted)} rows x {len(standard4)} coordinates, chunk={chunk}',flush=True)
 for hi in range(len(standard4)-1,-1,-chunk):
  lo=max(0,hi-chunk+1);indices=list(range(hi,lo-1,-1));fn=f'x36_profile_chunks/chunk_{hi:05d}_{lo:05d}.npy';blocks.append(fn)
  if os.path.exists(fn):
   M=np.load(fn);assert M.shape==(required,len(indices));continue
  if maxchunks and made>=maxchunks:break
  needed=len(indices);Proj=np.zeros((len(standard4),needed),dtype=np.uint64)
  for jj,ii in enumerate(indices):Proj[ii,jj]=1
  cache={};visiting=set();rows=[]
  for m,alt,k in accepted:
   v=(rewrite(m,alt).astype(np.int64)-nf(m).astype(np.int64))%p;rows.append(v.astype(np.uint32))
  M=np.array(rows,dtype=np.uint32);assert M.shape==(required,len(indices));np.save(fn,M);made+=1
  print(f'profile saved hi={hi} lo={lo} cache={len(cache)} nz={np.count_nonzero(M)} made={made} elapsed={time.time()-t:.1f}s',flush=True)
  del M,Proj,cache;gc.collect()
 if maxchunks and made>=maxchunks and any(not os.path.exists(fn) for fn in blocks):sys.exit(0)
 # Rebuild complete ordered block list, including any block names not reached because of maxchunks logic.
 blocks=[f'x36_profile_chunks/chunk_{hi:05d}_{max(0,hi-chunk+1):05d}.npy' for hi in range(len(standard4)-1,-1,-chunk)]
 assert all(os.path.exists(fn) for fn in blocks)
 all_desc=np.concatenate([np.load(fn) for fn in blocks],axis=1);assert all_desc.shape==(required,len(standard4))
 full=all_desc[:,::-1].copy();np.savez_compressed('x36_quartic_full_profiles.npz',relation_matrix=full)
 fh=hashlib.sha256(full.astype('<u4').tobytes()).hexdigest();print('FULL_PROFILE COMPLETE',full.shape,'nz',np.count_nonzero(full),'sha',fh,flush=True)
 open('x36_quartic_full_profiles_result.json','w').write(json.dumps({'prime':p,'shape':list(full.shape),'nonzeros':int(np.count_nonzero(full)),'matrix_sha256':fh,'artifact_sha256':hashlib.sha256(open('x36_quartic_full_profiles.npz','rb').read()).hexdigest()},indent=2,sort_keys=True))
 needed=required
res={'prime':p,'seed':seed,'geometric_h4':TARGET_H4,'quadratic_rules':len(qrules),'cubic_rules':len(crules),'standard_quartics_after_degree_2_3_leads':len(standard4),'required_quartic_residual_rank':needed,'quartic_confluence_rank':len(ebasis),'certified_quadratic_quotient_upper_bound':len(standard4)-len(ebasis),'residuals_tested':tested,'normal_forms_cached':len(cache),'projection_dimension':needed,'projection_sha256':ph,'accepted_residuals_sha256':ah,'projected_echelon_basis_sha256':bh.hexdigest(),'elapsed_seconds':time.time()-t}
open('x36_quartic_rank_result.json','w').write(json.dumps(res,indent=2,sort_keys=True));print(json.dumps(res,indent=2,sort_keys=True),flush=True)
if len(ebasis)<needed:sys.exit(2)
