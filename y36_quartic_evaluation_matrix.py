#!/usr/bin/env python3
# Exact F_p evaluation matrix for quartic canonical-function products on Naruki's Y(3,6).
# Selects 23730 of the 23817 standard quartic monomials by deleting the 87 certified
# quartic leading coordinates (whose 87x87 minor is invertible), then evaluates the
# selected monomials at 23730 deterministic points of the Naruki fourfold.
# A nonzero determinant certifies dim(S/I)_4 >= 23730 = h^0(4L), i.e. surjectivity of
# Sym^4 H^0(L) -> H^0(4L).
import gc,hashlib,json,os,random,sys,time
import numpy as np
sys.path.insert(0,os.getcwd()); import y36_cubic_sieve as y
p=1000003; seed_points=271828182; t=time.time()
D=np.load('y36_cubic_basis_data.npz'); coord=list(map(int,D['coord']))
standard4=np.load('standard4.npy'); keep=np.load('keep.npy')
assert standard4.shape==(23817,4) and keep.shape==(23730,)
idx=standard4[keep].astype(np.int64)
monhash=hashlib.sha256(standard4[keep].tobytes()).hexdigest()
nfull=23730; nrows=int(os.getenv('NROWS',str(nfull))); assert 1<=nrows<=nfull
outfile=os.getenv('OUTFILE','y36_quartic_eval_23730_u32.bin' if nrows==nfull else f'y36_quartic_eval_smoke_{nrows}_u32.bin')
rng=random.Random(seed_points)
points=np.empty((nrows,3,6),dtype=np.uint32)
M=np.memmap(outfile,dtype='<u4',mode='w+',shape=(nrows,nfull))
print(f'evaluating exact matrix {nrows}x{nfull}; monomial_sha256={monhash}; output={outfile}',flush=True)
for z in range(nrows):
    while True:
        X=[[rng.randrange(p) for _ in range(6)] for _ in range(3)]
        try:
            f=y.all_forms(X,p); break
        except ZeroDivisionError: pass
    points[z]=X
    a=np.asarray([f[i] for i in coord],dtype=np.uint64)
    v=(a[idx[:,0]]*a[idx[:,1]])%p
    v=(v*a[idx[:,2]])%p; v=(v*a[idx[:,3]])%p
    M[z,:]=v.astype(np.uint32)
    if (z+1)%1000==0 or z+1==nrows: print(f'rows {z+1}/{nrows} elapsed={time.time()-t:.1f}s',flush=True)
M.flush(); del M; gc.collect()
ph=hashlib.sha256(points.astype('<u4').tobytes()).hexdigest()
np.save(f'y36_quartic_eval_points{"" if nrows==nfull else f"_smoke_{nrows}"}.npy',points)
h=hashlib.sha256()
with open(outfile,'rb') as ff:
    while b:=ff.read(16<<20): h.update(b)
res={'prime':p,'point_seed':seed_points,'rows':nrows,'cols':nfull,'matrix_file':outfile,
     'matrix_sha256':h.hexdigest(),'points_sha256':ph,'selected_monomials_sha256':monhash,
     'elapsed_seconds':time.time()-t}
open(f'y36_quartic_eval_matrix_result{"" if nrows==nfull else "_smoke"}.json','w').write(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res,indent=2,sort_keys=True),flush=True)
