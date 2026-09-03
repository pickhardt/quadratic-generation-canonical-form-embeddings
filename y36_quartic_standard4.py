import itertools,os,sys,time,json
import numpy as np
from flint import nmod_mat
sys.path.insert(0,os.getcwd()); import y36_cubic_sieve as y
p=1000003; t=time.time()
D=np.load('y36_cubic_basis_data.npz')
coord=list(map(int,D['coord'])); perm=list(map(int,D['perm']))
pivs=set(map(int,D['quadratic_pivs']))
standard3=[tuple(map(int,x)) for x in D['standard_cubics']]
cnonp=list(map(int,D['cubic_nonp']))
assert len(coord)==150 and len(pivs)==1720 and len(standard3)==8582 and len(cnonp)==706
logical=y.grevlex_degree2_increasing(150)
mons=[tuple(sorted((perm[i],perm[j]))) for i,j in logical]
qleads={mons[j] for j in range(len(mons)) if j not in pivs}
cleads={standard3[j] for j in cnonp}
print(f'qleads={len(qleads)} cleads={len(cleads)}',flush=True)

standard4=[]
for m in itertools.combinations_with_replacement(range(150),4):
    if any((m[i],m[j]) in qleads for i in range(4) for j in range(i+1,4)): continue
    if any(tuple(m[z] for z in range(4) if z!=o) in cleads for o in range(4)): continue
    standard4.append(m)
print(f'standard4={len(standard4)}  (paper: 23817)  elapsed={time.time()-t:.1f}s',flush=True)
assert len(standard4)==23817

R=np.load('y36_quartic_oriented_rules.npz')
leads=list(map(int,R['leads'])); rules=R['rules']
assert len(leads)==87 and rules.shape==(87,23817)
minor=rules[:,leads]
det=int(nmod_mat(minor.astype(np.int64).tolist(),p).det())
print('87x87 lead minor det mod p =',det,' (nonzero required)',flush=True)
assert det!=0
keep=[j for j in range(23817) if j not in set(leads)]
assert len(keep)==23730
np.save('standard4.npy',np.array(standard4,dtype=np.uint8))
np.save('keep.npy',np.array(keep,dtype=np.int64))
json.dump({'standard4':len(standard4),'leads':len(leads),'keep':len(keep),
           'lead_minor_det_mod_p':det},open('standard4_result.json','w'),indent=2)
print('OK: keep =',len(keep),'= h^0(4L) = 23730',flush=True)
