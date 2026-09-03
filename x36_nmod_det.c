// Exact determinant over F_1000003 of a uint32 row-major matrix file.
// Built against the FLINT library bundled with python-flint 0.9.0 (FLINT 3.6.0).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <time.h>
typedef long slong; typedef unsigned long ulong;
typedef struct { ulong n,ninv,norm; } nmod_t;
typedef struct { ulong *entries; slong r,c,stride; nmod_t mod; } nmod_mat_struct;
typedef nmod_mat_struct nmod_mat_t[1];
extern void nmod_mat_init(nmod_mat_t, slong, slong, ulong);
extern void nmod_mat_clear(nmod_mat_t);
extern ulong nmod_mat_det(const nmod_mat_t);
static atomic_int running=1; static double started;
static double nowsec(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void *heartbeat(void *x){(void)x;int ticks=0;while(atomic_load(&running)){sleep(1);ticks++;if(atomic_load(&running)&&ticks%30==0){printf("heartbeat determinant elapsed=%.0fs\n",nowsec()-started);fflush(stdout);}}return NULL;}
int main(int argc,char **argv){
 if(argc<2){fprintf(stderr,"usage: %s matrix.bin [n] [file_cols]\n",argv[0]);return 2;}
 const char *fn=argv[1]; slong n=argc>2?atol(argv[2]):18695; slong fc=argc>3?atol(argv[3]):18695; const ulong p=1000003;
 struct stat st;if(stat(fn,&st)){perror("stat");return 2;} long long expected=4LL*fc*fc;
 if(st.st_size<4LL*n*fc){fprintf(stderr,"file too short: %lld need %lld\n",(long long)st.st_size,4LL*n*fc);return 2;}
 printf("loading exact nmod matrix n=%ld file_cols=%ld bytes=%lld expected_full=%lld\n",n,fc,(long long)st.st_size,expected);fflush(stdout);
 FILE *f=fopen(fn,"rb");if(!f){perror("fopen");return 2;} uint32_t *row=malloc(4*(size_t)fc);if(!row){perror("malloc row");return 2;}
 nmod_mat_t A;nmod_mat_init(A,n,n,p);started=nowsec();
 for(slong i=0;i<n;i++){if(fread(row,4,fc,f)!=(size_t)fc){fprintf(stderr,"short read row %ld\n",i);return 2;}ulong *dst=A->entries+i*A->stride;for(slong j=0;j<n;j++)dst[j]=row[j];if((i+1)%2000==0){printf("loaded %ld/%ld elapsed=%.1fs\n",i+1,n,nowsec()-started);fflush(stdout);}}
 free(row);fclose(f);printf("load complete elapsed=%.1fs; starting FLINT nmod_mat_det\n",nowsec()-started);fflush(stdout);
 pthread_t th;pthread_create(&th,NULL,heartbeat,NULL);double td=nowsec();ulong det=nmod_mat_det(A);double ed=nowsec()-td;atomic_store(&running,0);pthread_join(th,NULL);
 printf("DETERMINANT_MOD_P=%lu\n",det);printf("det_seconds=%.6f total_seconds=%.6f\n",ed,nowsec()-started);fflush(stdout);nmod_mat_clear(A);return det?0:1;
}
