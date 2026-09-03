#!/usr/bin/env python3
# Orient the exact 81-dimensional quartic relation space at genuine grevlex leads.
import hashlib,json,numpy as np,time
p=1000003;t=time.time()
R=np.load('x36_quartic_full_profiles.npz')['relation_matrix'].astype(np.uint64)
meta=json.load(open('x36_quartic_eval_matrix_result.json'))
leads=np.array(meta['omitted_standard4_indices'],dtype=int)
assert R.shape==(81,18776) and leads.shape==(81,)
A=R[:,leads].copy()%p
Aug=np.concatenate([A,np.eye(81,dtype=np.uint64)],axis=1)
for j in range(81):
 nz=np.flatnonzero(Aug[j:,j]);assert len(nz)
 k=j+int(nz[0]);Aug[[j,k]]=Aug[[k,j]]
 Aug[j]=(Aug[j]*np.uint64(pow(int(Aug[j,j]),p-2,p)))%p
 for i in range(81):
  if i==j:continue
  a=int(Aug[i,j])
  if a:Aug[i]=(Aug[i]+np.uint64(p-a)*Aug[j])%p
C=Aug[:,81:];G=(C@R)%p
assert np.array_equal(G[:,leads],np.eye(81,dtype=np.uint64))
assert all(np.flatnonzero(G[i])[-1]==leads[i] for i in range(81))
np.savez_compressed('x36_quartic_oriented_rules.npz',leads=leads.astype(np.int32),rules=G.astype(np.uint32),change_of_basis=C.astype(np.uint32))
def fh(fn):return hashlib.sha256(open(fn,'rb').read()).hexdigest()
res={'prime':p,'raw_shape':list(R.shape),'raw_nonzeros':int(np.count_nonzero(R)),'change_of_basis_nonzeros':int(np.count_nonzero(C)),'oriented_nonzeros':int(np.count_nonzero(G)),'raw_matrix_sha256':hashlib.sha256(R.astype('<u4').tobytes()).hexdigest(),'oriented_matrix_sha256':hashlib.sha256(G.astype('<u4').tobytes()).hexdigest(),'lead_indices_sha256':hashlib.sha256(leads.astype('<u4').tobytes()).hexdigest(),'artifact_sha256':fh('x36_quartic_oriented_rules.npz'),'elapsed_seconds':time.time()-t}
open('x36_orient_quartic_rules_result.json','w').write(json.dumps(res,indent=2,sort_keys=True));print(json.dumps(res,indent=2,sort_keys=True))
