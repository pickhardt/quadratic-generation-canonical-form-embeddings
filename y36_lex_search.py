#!/usr/bin/env python3
import gc,json,random,time,sys,os
sys.path.insert(0,os.getcwd())
from flint import fmpz_mod_ctx,fmpz_mod_mat
import y36_cubic_sieve as y
p=1000003;rng=random.Random(260728368);ctx=fmpz_mod_ctx(p);t=time.time()
coord=y.pivot_columns(fmpz_mod_mat([y.random_point(rng,p) for _ in range(170)],ctx));assert len(coord)==150
Y=[]
for z in range(1720):
 f=y.random_point(rng,p);Y.append([f[i] for i in coord])
def lexinc(perm):
 # descending lex physical/logical list is i=0.., j=i..; reverse for increasing.
 return [tuple(sorted((perm[i],perm[j]))) for i in range(149,-1,-1) for j in range(149,i-1,-1)]
tr=random.Random(13579);perms=[list(range(150)),list(reversed(range(150)))]
for z in range(8):
 a=list(range(150));tr.shuffle(a);perms.append(a)
for trial,perm in enumerate(perms):
 mons=lexinc(perm);qrows=[[(r[i]*r[j])%p for i,j in mons] for r in Y]
 Q=fmpz_mod_mat(qrows,ctx);del qrows;pivs=y.pivot_columns(Q);del Q;gc.collect()
 c=y.cubic_standard_count([mons[i] for i in pivs],150)
 print(json.dumps({'trial':trial,'perm_sha256':y.sha_ints(perm),'rank':len(pivs),'standard3':c,'shadow':573800-c,'piv_sha256':y.sha_ints(pivs)},sort_keys=True),'elapsed',time.time()-t,flush=True)
