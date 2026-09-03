#!/usr/bin/env python3
import gc,itertools,json,random,time
from flint import fmpz_mod_ctx,fmpz_mod_mat
import y36_cubic_sieve as y
p=1000003; seed=260728368; rng=random.Random(seed); ctx=fmpz_mod_ctx(p); t=time.time()
rows=[y.random_point(rng,p) for _ in range(170)]
coord=y.pivot_columns(fmpz_mod_mat(rows,ctx))
assert len(coord)==150
Y=[]
for z in range(1720):
 f=y.random_point(rng,p); Y.append([f[i] for i in coord])
 if (z+1)%200==0: print('points',z+1,'elapsed',time.time()-t,flush=True)
def mon_order(perm):
 # logical variables perm[0],...,perm[149]; smallest-to-largest grevlex.
 logical=y.grevlex_degree2_increasing(150)
 return [tuple(sorted((perm[i],perm[j]))) for i,j in logical]
trng=random.Random(8675309)
perms=[list(range(150)),list(reversed(range(150)))]
for z in range(18):
 a=list(range(150));trng.shuffle(a);perms.append(a)
out=[]
for trial,perm in enumerate(perms):
 mons=mon_order(perm)
 qrows=[[(r[i]*r[j])%p for i,j in mons] for r in Y]
 Q=fmpz_mod_mat(qrows,ctx); del qrows
 piv=y.pivot_columns(Q); del Q; gc.collect()
 pp=[mons[i] for i in piv]; c=y.cubic_standard_count(pp,150)
 rec={'trial':trial,'perm_sha256':y.sha_ints(perm),'rank':len(piv),'standard3':c,'shadow':573800-c,'piv_sha256':y.sha_ints(piv)}
 out.append(rec); print(json.dumps(rec,sort_keys=True),'elapsed',time.time()-t,flush=True)
 open('y36_termorder_results.json','w').write(json.dumps(out,indent=2,sort_keys=True))
