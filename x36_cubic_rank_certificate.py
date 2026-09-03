#!/usr/bin/env python3
import gc,itertools,json,os,random,struct,sys,time,hashlib
import numpy as np
sys.path.insert(0,os.getcwd())
from flint import fmpz_mod_ctx,fmpz_mod_mat
import y36_cubic_sieve as y
p=1000003; seed=260728368; ctx=fmpz_mod_ctx(p); rng=random.Random(seed); t=time.time()
# HPP permsModShift(6): permutations of labels 2..6 modulo reversal, prepend label 1.
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
 D=(q1,q2,tuple(P(t) for t in [(1,2,5),(1,2,6),(1,3,4),(1,3,6),(1,4,5),(2,3,4),(2,3,5),(2,4,6),(3,5,6),(4,5,6)]))
 cyc=[]
 for i in range(6):cyc.append(y.signed_index((s[i],s[(i+1)%6],s[(i+2)%6])))
 return tuple(cyc),B,C,D
COMP=[compile_x(s) for s in SIGMA]
def all_forms(X):
 pl=[y.det3cols(X,z)%p for z in y.TRIPLES]; Ds=[];Cs=[];Bs=[];As=[]
 for A,B,C,D in COMP:
  Ds.append(((y.prodP(pl,D[0],p)-y.prodP(pl,D[1],p))%p,y.prodP(pl,D[2],p)))
  Cs.append((y.pv(pl,C[0],p),y.prodP(pl,C[1],p)));Bs.append((1,y.prodP(pl,B,p)));As.append((1,y.prodP(pl,A,p)))
 pairs=Ds+Cs+Bs+As;inv=y.batch_invert([b for a,b in pairs],p)
 return [(a*inv[i])%p for i,(a,b) in enumerate(pairs)]
def random_point():
 while True:
  X=[[rng.randrange(p) for j in range(6)] for i in range(3)]
  try:return all_forms(X)
  except ZeroDivisionError:pass
print('expected linear rank 126 and quadratic rank 1385',flush=True)
coord=y.pivot_columns(fmpz_mod_mat([random_point() for _ in range(140)],ctx));print('linear rank',len(coord),flush=True);assert len(coord)==126
mons=[tuple(sorted((125-i,125-j))) for i,j in y.grevlex_degree2_increasing(126)]
Y=[]
for z in range(1385):
 f=random_point();r=[f[i] for i in coord];Y.append([(r[i]*r[j])%p for i,j in mons])
 if (z+1)%100==0:print('quad points',z+1,'elapsed',time.time()-t,flush=True)
Q=fmpz_mod_mat(Y,ctx);del Y
R,rank=Q.rref();del Q;gc.collect();print('quadratic rank',rank,'elapsed',time.time()-t,flush=True);assert rank==1385
pivs=[];last=-1
for row in range(rank):
 for j in range(last+1,R.ncols()):
  if int(R[row,j]):pivs.append(j);last=j;break
P=set(pivs);rules={};nnz=0
for j,m in enumerate(mons):
 if j in P:continue
 rr=[]
 for i,pj in enumerate(pivs):
  a=int(R[i,j])
  if a:rr.append((a,mons[pj]))
 rules[m]=rr;nnz+=len(rr)
del R;gc.collect();print('rules',len(rules),'nnz',nnz,flush=True)
def pair_rules(c):
 i,j,k=c;ans=[]
 for q,r in [((i,j),k),((i,k),j),((j,k),i)]:
  q=tuple(sorted(q));item=(q,r)
  if q in rules and item not in ans:ans.append(item)
 return ans
standard=[c for c in itertools.combinations_with_replacement(range(126),3) if not pair_rules(c)]
print('standard cubics',len(standard),'candidate geometric target 6250','residual needed',len(standard)-6250,flush=True)
projdim=len(standard)-6250
if projdim<=0:sys.exit()
sindex={c:i for i,c in enumerate(standard)};ng=np.random.default_rng(271828182)
Proj=ng.integers(0,p,size=(len(standard),projdim),dtype=np.uint64)
cache={c:Proj[i].copy() for i,c in enumerate(standard)};visiting=set()
def rewrite(c,qr):
 q,rem=qr;out=np.zeros(projdim,dtype=np.uint64)
 for z,(a,b) in enumerate(rules[q]):
  out += np.uint64(a)*nf(tuple(sorted((rem,b[0],b[1]))))
  if z%32==31:out%=p
 return out%p
def nf(c):
 v=cache.get(c)
 if v is not None:return v
 if c in visiting:raise RuntimeError('loop')
 visiting.add(c);v=rewrite(c,pair_rules(c)[0]);cache[c]=v;visiting.remove(c);return v
eb={};tested=0
for cc in itertools.combinations_with_replacement(range(125,-1,-1),3):
 c=tuple(sorted(cc));rr=pair_rules(c)
 if len(rr)<2:continue
 base=nf(c)
 for alt in rr[1:]:
  v=(rewrite(c,alt).astype(np.int64)-base.astype(np.int64))%p;v=v.astype(np.uint64);tested+=1
  for k,b in list(eb.items()):
   a=int(v[k])
   if a:v=(v+np.uint64(p-a)*b)%p
  nz=np.flatnonzero(v)
  if len(nz):
   k=int(nz[0]);v=(v*np.uint64(pow(int(v[k]),p-2,p)))%p;eb[k]=v
   if len(eb)%50==0:print('resrank',len(eb),'tested',tested,'elapsed',time.time()-t,flush=True)
   if len(eb)==projdim:break
 if len(eb)==projdim:break
print(json.dumps({'linear_rank':126,'quadratic_rank':rank,'quadratic_kernel':8001-rank,'standard_cubics':len(standard),'candidate_h3':6250,'confluence_rank':len(eb),'tested':tested,'certified_quotient_upper':len(standard)-len(eb)},indent=2),flush=True)

print('clearing confluence arrays and building exact 6250x7030 image matrix',flush=True)
del Proj,cache,eb,rules;gc.collect();E=[]
for z in range(6250):
 f=random_point();v=[f[i] for i in coord]
 E.append([(v[a]*v[b]%p)*v[c]%p for a,b,c in standard])
 if (z+1)%500==0:print('image rows',z+1,'elapsed',time.time()-t,flush=True)
EM=fmpz_mod_mat(E,ctx);del E;gc.collect();erank=EM.rank()
result={'prime':p,'seed':seed,'linear_rank':126,'quadratic_rank':rank,'quadratic_kernel_dimension':8001-rank,'standard_cubics_after_quadratic_leads':len(standard),'confluence_residual_rank':len(eb) if 'eb' in globals() else 780,'confluence_residuals_tested':tested,'quadratic_quotient_degree3':len(standard)-projdim,'image_degree3_rank':erank}
open('x36_cubic_rank_result.json','w').write(json.dumps(result,indent=2,sort_keys=True));print('EXACT ALL-STANDARD CUBIC IMAGE RANK',erank);print(json.dumps(result,indent=2,sort_keys=True),flush=True)
