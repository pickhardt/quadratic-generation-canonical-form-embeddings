import re,pathlib,itertools,collections,numpy as np
from scipy.sparse import coo_matrix,vstack
from scipy.sparse.linalg import lsqr
P=1000003
text=pathlib.Path('ChowM36.sage').read_text()
m=re.search(r'R\.<(.*?)> = QQ\[\]',text,re.S)
Ds=[x.strip() for x in m.group(1).replace('\n','').split(',') if x.strip().startswith('D')]
idx={x:i for i,x in enumerate(Ds)}
def typ(x):return tuple(map(len,x[1:].split('_')))
def short(x):return x.split('_')[0] if typ(x) in [(3,3),(2,4)] else x
mp={short(x):i for i,x in enumerate(Ds)}
# parse all simple positive sum assignments
expr={}
for line in text.splitlines():
 mm=re.match(r'^(\w+)\s*=\s*(.+)$',line.strip())
 if mm and all(c not in mm.group(2) for c in '[]()'):
  name,rhs=mm.groups();terms=[x.strip() for x in rhs.split('+')]
  if all(re.match(r'^\w+$',x) for x in terms):
   c=collections.defaultdict(int);ok=True
   for x in terms:
    if x in idx:c[idx[x]]+=1
    elif x in expr:
     for y,v in expr[x].items():c[y]+=v
    else:ok=False
   if ok:expr[name]=dict(c)
def add(*cs):
 o=collections.defaultdict(int)
 for coef,c in cs:
  for x,v in c.items():o[x]+=coef*v
 return {x:v for x,v in o.items() if v}
def rv(mm,i,j):
 if i>j:i,j=j,i
 return expr[f'r{mm}_{i}{j}']
# all linear rels
rels=[]
for i,j,k,l in itertools.permutations(range(1,7),4):
 for mm in range(1,7):
  if mm not in [i,j,k,l]:rels.append(add((1,rv(mm,i,j)),(1,rv(mm,k,l)),(-1,rv(mm,i,l)),(-1,rv(mm,j,k))))
# select modular independent rels
piv={}; indep=[]
for rr in rels:
 v={i:a%P for i,a in rr.items() if a%P}
 while v:
  q=max(v)
  if q not in piv:
   inv=pow(v[q],P-2,P);v={i:(a*inv)%P for i,a in v.items()};piv[q]=v;indep.append(rr);break
  a=v[q];b=piv[q]
  for i,z in b.items():
   v[i]=(v.get(i,0)-a*z)%P
   if not v[i]:v.pop(i,None)
print('linear rank',len(indep),flush=True)
# edges
labels=set(range(1,7));comb=itertools.combinations;E=list(map(frozenset,comb(labels,3)));F=list(map(frozenset,comb(labels,2)))
raw=[((1,2),(3,4),(5,6)),((1,6),(2,3),(4,5)),((1,5),(2,3),(4,6)),((1,4),(2,3),(5,6)),((1,3),(2,4),(5,6)),((1,3),(2,5),(4,6)),((1,4),(2,5),(3,6)),((1,5),(2,4),(3,6)),((1,6),(2,4),(3,5)),((1,6),(2,5),(3,4)),((1,5),(2,6),(3,4)),((1,4),(2,6),(3,5)),((1,3),(2,6),(4,5)),((1,2),(3,6),(4,5)),((1,2),(3,5),(4,6))]
G=[]
for a,b,c in raw:G += [(frozenset(a),frozenset(b),frozenset(c)),(frozenset(b),frozenset(a),frozenset(c))]
def lab(s):return ''.join(map(str,sorted(s)))
def glab(g):return '_'.join(lab(x) for x in g)
edges=set()
for a,b in comb(E,2):
 if len(a&b) in [0,1]:edges.add(frozenset(['D'+lab(a),'D'+lab(b)]))
for a,b in comb(F,2):
 if len(a&b)==0:edges.add(frozenset(['D'+lab(a),'D'+lab(b)]))
for e in E:
 for f in F:
  if len(e&f) in [0,2]:edges.add(frozenset(['D'+lab(e),'D'+lab(f)]))
for f in F:
 for g in G:
  if any(len(f&x)==2 for x in g):edges.add(frozenset(['D'+lab(f),'D'+glab(g)]))
def valid(e,g):return any(len(e&g[i])==2 and len(e&g[(i+1)%3])==1 and len(e&g[(i+2)%3])==0 for i in range(3))
for e in E:
 for g in G:
  if valid(e,g):edges.add(frozenset(['D'+lab(e),'D'+glab(g)]))
adj=[[True]*65 for _ in range(65)]
for i in range(65):
 for j in range(i+1,65):adj[i][j]=adj[j][i]=frozenset([short(Ds[i]),short(Ds[j])]) in edges
def allowed(mon):return all(adj[i][j] for i,j in comb(set(mon),2))
mons={d:[z for z in itertools.combinations_with_replacement(range(65),d) if allowed(z)] for d in [3,4]}
mi={z:i for i,z in enumerate(mons[4])};print('mon counts',len(mons[3]),len(mons[4]),flush=True)
rows=[];cols=[];data=[];nr=0
for base in mons[3]:
 for rr in indep:
  row=collections.defaultdict(int)
  for i,a in rr.items():
   z=tuple(sorted(base+(i,)))
   if allowed(z):row[mi[z]]+=a
  row={i:a for i,a in row.items() if a}
  if row:
   for i,a in row.items():rows.append(nr);cols.append(i);data.append(float(a))
   nr+=1
A=coo_matrix((data,(rows,cols)),shape=(nr,len(mons[4]))).tocsr();print('A',A.shape,A.nnz,flush=True)
# polynomial power coefficient vector sparse exact integers
def linpow(c,d=4):
 poly={():1}
 for _ in range(d):
  out=collections.defaultdict(int)
  for mon,a in poly.items():
   for i,b in c.items():
    z=tuple(sorted(mon+(i,)))
    if allowed(z):out[z]+=a*b
  poly={z:a for z,a in out.items() if a}
 return poly
def dense(poly):
 x=np.zeros(len(mons[4]))
 for z,a in poly.items():x[mi[z]]=a
 return x
H1=add((1,rv(6,1,2)),(1,rv(6,3,5)),(1,rv(6,4,5)))
H2=add((1,rv(5,1,2)),(1,rv(5,3,6)),(1,rv(5,4,6)))
# mixed H1^2 H2^2
p1=linpow(H1,2);p2=linpow(H2,2)
ref=collections.defaultdict(int)
for a,ca in p1.items():
 for b,cb in p2.items():
  z=tuple(sorted(a+b))
  if allowed(z):ref[z]+=ca*cb
refv=dense(ref)
# append normalization and solve A w=0, ref.w=1
AA=vstack([A,coo_matrix(refv.reshape(1,-1))]).tocsr();bb=np.zeros(AA.shape[0]);bb[-1]=1
sol=lsqr(AA,bb,atol=1e-13,btol=1e-13,iter_lim=100000,show=False)
w0=sol[0]; w=np.rint(w0).astype(np.int64); assert np.max(np.abs(w0-w))<1e-6; assert np.max(np.abs(A@w))==0; assert int(np.dot(w,refv))==1; print('exact rounded functional verified');print('lsqr status',sol[1:5],'range',w.min(),w.max(),flush=True)
# KB=(7 B33+8 B24+12 B222)/10
kb10={i:(7 if typ(x)==(3,3) else 8 if typ(x)==(2,4) else 12) for i,x in enumerate(Ds)}
kbpoly=linpow(kb10,4);kbval=float(np.dot(w,dense(kbpoly)))/10000
print('KB4 approx',kbval,'ref',np.dot(w,refv),'resid max',np.max(np.abs(A@w)),flush=True)
# write w and values
np.savez_compressed('x36_chow_degree_certificate.npz',w=w,ref=refv,kb=dense(kbpoly))
import json,hashlib
def multlin(poly,c):
 out=collections.defaultdict(int)
 for mon,a in poly.items():
  for i,b in c.items():
   z=tuple(sorted(mon+(i,)))
   if allowed(z):out[z]+=a*b
 return dict(out)
def mixed(classes):
 poly={():1}
 for c in classes:poly=multlin(poly,c)
 return sum(a*int(w[mi[z]]) for z,a in poly.items())
# 10*c1(TX)=-10*K=3B_33+2B_24-2B_222, from Schock Prop. 7.1.
c110={i:(3 if typ(x)==(3,3) else 2 if typ(x)==(2,4) else -2) for i,x in enumerate(Ds)}
nums={'L_four':mixed([kb10]*4),'L3_c1':mixed([kb10]*3+[c110]),'L2_c1sq':mixed([kb10]*2+[c110]*2),'L_c1cube':mixed([kb10]+[c110]*3),'c1_four':mixed([c110]*4)}
assert all(v%10000==0 for v in nums.values())
res={'linear_relation_rank':len(indep),'degree3_allowed_monomials':len(mons[3]),'degree4_allowed_monomials':len(mons[4]),'degree4_relation_rows':A.shape[0],'degree4_relation_nnz':A.nnz,'functional_min':int(w.min()),'functional_max':int(w.max()),'reference_degree':int(np.dot(w,refv)),'tenL_four_numerator':nums['L_four'],**{k:v//10000 for k,v in nums.items()}}
open('x36_chow_degree_result.json','w').write(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res,indent=2,sort_keys=True))


from collections import defaultdict
def padd(*terms):
 o=defaultdict(int)
 for q,a in terms:
  for m,c in a.items():o[m]+=q*c
 return {m:c for m,c in o.items() if c}
def pmul(a,b):
 o=defaultdict(int)
 for x,c in a.items():
  for y,d in b.items():
   z=tuple(sorted(x+y))
   if allowed(z):o[z]+=c*d
 return {m:c for m,c in o.items() if c}
def lin(a):return {(i,):c for i,c in a.items() if c}
def lv(s):return {idx[s]:1}
def lsum(a):return add(*[(1,x) for x in a]) if a else {}
def lp(a,b):return pmul(lin(a),lin(b))
def lsq(a):return lp(a,a)
EA=[rv(6,i,5) for i in range(1,5)];EB=[rv(5,i,6) for i in range(1,5)]
c1a=add((3,H1),(-1,lsum(EA)));c1b=add((3,H2),(-1,lsum(EB)))
c1=add((1,c1a),(1,c1b));c2=padd((7,lsq(H1)),(7,lsq(H2)),(1,lp(c1a,c1b)))
e2=[]
for l in range(1,5):
 q=sorted(set(range(1,5))-{l});e=lv('D'+''.join(map(str,q))+'_'+str(l)+'56');e2.append(e)
 z=lp(rv(6,l,5),rv(5,l,6));c2=padd((1,c2),(-1,pmul(lin(c1),lin(e))),(1,z));c1=add((1,c1),(-1,e))
order=[(1,2),(1,3),(2,3),(1,4),(2,4),(3,4)];e3=[];extra=[]
for i,j in order:
 q=sorted(set(range(1,5))-{i,j});a=f'{i}{j}';b=''.join(map(str,q));x=lv(f'D{a}_{b}_56');e=add((1,lv(f'D{a}_{b}56')),(1,x));extra.append(x);e3.append(e)
 z=lp(rv(6,i,j),rv(5,i,j));c2=padd((1,c2),(-1,pmul(lin(c1),lin(e))),(1,z));c1=add((1,c1),(-1,e))
D2=lsum(e2);D31=lsum(e3[:3]);D32=lsum(e3[3:]);D3=add((1,D31),(1,D32))
z4=padd((1,lsq(H1)),(1,lp(H1,H2)),(1,lsq(H2)),(1,lsq(D2)),(1,lp(D31,D32)),(-1,lp(H1,add((1,D2),(1,D3)))),(-1,lp(H2,add((1,D2),(1,D3)))),(1,lp(D2,D3)))
e4=lv('D56_1234')
c2=padd((1,c2),(-1,pmul(lin(c1),lin(e4))),(1,z4));c1=add((1,c1),(-1,e4))
def d3name(T):
 T=set(T);C=set(range(1,7))-T
 s='D'+''.join(map(str,sorted(T)))+'_'+''.join(map(str,sorted(C)))
 if s not in idx: raise KeyError(s)
 return s
for s in [x for x in Ds if typ(x)==(2,2,2)]:
 g=s[1:].split('_');g2=list(map(int,g[1]));g3=list(map(int,g[2]));E=lv(s);A=lv(d3name(g2+[g3[0]]));B=lv('D'+g[2]+'_'+''.join(map(str,sorted(set(range(1,7))-set(g3)))))
 z=lp(add((1,A),(1,E)),add((1,B),(1,E)))
 c2=padd((1,c2),(-1,pmul(lin(c1),lin(E))),(1,z));c1=add((1,c1),(-1,E))
# c1 test mod linear relations
v={i:(10*c1.get(i,0)-c110.get(i,0))%P for i in range(65) if (10*c1.get(i,0)-c110.get(i,0))%P}
while v:
 q=max(v)
 if q not in piv:break
 a=v[q]
 for i,z in piv[q].items():
  v[i]=(v.get(i,0)-a*z)%P
  if not v[i]:v.pop(i,None)
print('C1_REMAINDER',v)
def deg(poly):return sum(a*int(w[mi[m]]) for m,a in poly.items())
num=deg(pmul(pmul(lin(kb10),lin(kb10)),c2))
print('L2C2_NUM100',num,'L2C2',num/100)
print('C1SQ_C2',deg(pmul(lsq(c1),c2)))
