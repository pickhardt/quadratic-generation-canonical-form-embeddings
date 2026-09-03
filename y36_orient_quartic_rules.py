#!/usr/bin/env python3
# Orient the exact 87-dimensional quartic relation space at its genuine
# grevlex leading coordinates over F_1000003.
import hashlib,json,numpy as np,time
p=1000003;t=time.time()
R=np.load('y36_quartic_full_profiles.npz')['relation_matrix'].astype(np.uint64)
leads=np.load('y36_quartic_profiles.npz')['q4lead_indices'].astype(int)
assert R.shape==(87,23817) and leads.shape==(87,)
A=R[:,leads].copy()%p
Aug=np.concatenate([A,np.eye(87,dtype=np.uint64)],axis=1)
for j in range(87):
 nz=np.flatnonzero(Aug[j:,j]);assert len(nz)
 k=j+int(nz[0]);Aug[[j,k]]=Aug[[k,j]]
 Aug[j]=(Aug[j]*np.uint64(pow(int(Aug[j,j]),p-2,p)))%p
 for i in range(87):
  if i==j:continue
  a=int(Aug[i,j])
  if a:Aug[i]=(Aug[i]+np.uint64(p-a)*Aug[j])%p
C=Aug[:,87:];G=(C@R)%p
assert np.array_equal(G[:,leads],np.eye(87,dtype=np.uint64))
assert all(np.flatnonzero(G[i])[-1]==leads[i] for i in range(87))
np.savez_compressed('y36_quartic_oriented_rules.npz',leads=leads.astype(np.int32),rules=G.astype(np.uint32),change_of_basis=C.astype(np.uint32))
def fh(fn):return hashlib.sha256(open(fn,'rb').read()).hexdigest()
res={'prime':p,'raw_shape':list(R.shape),'raw_nonzeros':int(np.count_nonzero(R)),'change_of_basis_nonzeros':int(np.count_nonzero(C)),'oriented_nonzeros':int(np.count_nonzero(G)),'oriented_matrix_sha256':hashlib.sha256(G.astype('<u4').tobytes()).hexdigest(),'artifact_sha256':fh('y36_quartic_oriented_rules.npz'),'elapsed_seconds':time.time()-t}
open('y36_orient_quartic_rules_result.json','w').write(json.dumps(res,indent=2,sort_keys=True));print(json.dumps(res,indent=2,sort_keys=True))
