/* Routines for partition convolutions */

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
#include <time.h> 
#include <limits.h> 
#include <assert.h> 
#include <valgrind/callgrind.h> 

#ifdef FFT_LIB_FFTW

#include <complex.h>
#include <fftw3.h> 

#define FFTW_MALLOC(n) (fftw_complex*)fftw_malloc((n)*sizeof(fftw_complex))
#define FFTW_FREE_(x) if (x) { fftw_free(x); x = NULL; } 

typedef fftw_complex c64_t;

#endif  

typedef double f64_t;

#define FREE_(x)      if (x) { free(x); x = NULL; } 
#define CALLOC(t,n) (t*)calloc(sizeof(t),n) 
#define CONT_IF_NULL(x,flg) if (!x) { flg = 1; continue; }
#define BRK_IF_NULL(x,flg,cond) if (!x) { cond; flg = 1; continue; }
#define RAND_C64() (((c64_t)random())/LONG_MAX * 2. - 1.)

static inline void *MCHK (void *x)
{
    if (!x) {
        fprintf(stderr,"Out of memory.");
        abort();
    }
    return x;
}

typedef enum err_t {
    err_NONE,
    err_EINVAL,
    err_MEM,
} err_t;

typedef struct part_conv_t {
    /* Size of input vector */
    size_t M;
    /* Size of impulse response */
    size_t N;
    /* Number of partitions */
    size_t d;
    /* Logical size of parts i.e., the next power of 2 */
    size_t _parts_sz;
    /* The IR partitions */
    c64_t **ir_parts;
    /* Buffers for processing */
    c64_t **tmp_bufs;
#ifdef FFT_LIB_FFTW
    fftw_plan *_ir_part_plans; 
    fftw_plan *_tmp_bufs_plans_fwd; 
    fftw_plan *_tmp_bufs_plans_bwd; 
#endif  
} part_conv_t;

static f64_t next_pow_2 (float x)
{
    return pow(2.,ceil(log(x)/log(2.)));
}

err_t
part_conv_init (part_conv_t *pc, size_t M, size_t N, size_t d)
{
    if ((d < 1) || (M < 1) || (N < 1) || (N % d)) {
        return err_EINVAL;
    }

    pc->ir_parts = NULL;
    pc->tmp_bufs = NULL;
    pc->_ir_part_plans = NULL; 
    pc->_tmp_bufs_plans_fwd = NULL; 
    pc->_tmp_bufs_plans_bwd = NULL; 
    pc->_parts_sz = next_pow_2(M+N/d-1);
    int me = 0;
    do {
        pc->ir_parts = CALLOC(fftw_complex*,d);
        CONT_IF_NULL(pc->ir_parts,me);
        pc->tmp_bufs = CALLOC(fftw_complex*,d);
        CONT_IF_NULL(pc->tmp_bufs,me);
        pc->_ir_part_plans = CALLOC(fftw_plan,d);
        CONT_IF_NULL(pc->_ir_part_plans,me);
        pc->_tmp_bufs_plans_fwd = CALLOC(fftw_plan,d);
        CONT_IF_NULL(pc->_tmp_bufs_plans_fwd,me);
        pc->_tmp_bufs_plans_bwd = CALLOC(fftw_plan,d);
        CONT_IF_NULL(pc->_tmp_bufs_plans_bwd,me);
        do {
            size_t _d;
            for (_d = 0; _d < d; _d++) {
#ifdef FFT_LIB_FFTW 
                pc->ir_parts[_d] = FFTW_MALLOC(pc->_parts_sz/2+1);
                BRK_IF_NULL(pc->ir_parts[_d],me,_d=d);
                pc->_ir_part_plans[_d] = fftw_plan_dft_r2c_1d(pc->_parts_sz,
                                                              (double*)pc->ir_parts[_d],
                                                              pc->ir_parts[_d],
                                                              FFTW_ESTIMATE);
                pc->tmp_bufs[_d] = FFTW_MALLOC(pc->_parts_sz/2+1);
                BRK_IF_NULL(pc->tmp_bufs[_d],me,_d=d);
                pc->_tmp_bufs_plans_fwd[_d] = fftw_plan_dft_r2c_1d(pc->_parts_sz,
                                                              (double*)pc->tmp_bufs[_d],
                                                              pc->tmp_bufs[_d],
                                                              FFTW_ESTIMATE);
                pc->_tmp_bufs_plans_bwd[_d] = fftw_plan_dft_c2r_1d(pc->_parts_sz,
                                                              pc->tmp_bufs[_d],
                                                              (double*)pc->tmp_bufs[_d],
                                                              FFTW_ESTIMATE);
#endif  
            }
        } while (0);
        if (me) {
            size_t _d;
            for ( _d = 0 ; _d < d; _d++) {
#ifdef FFT_LIB_FFTW 
                FFTW_FREE_(pc->ir_parts[_d]);
                FFTW_FREE_(pc->tmp_bufs[_d]);
#endif  
            }
        }
    } while (0);
    if (me) {
        FREE_(pc->ir_parts);
        FREE_(pc->tmp_bufs);
#ifdef FFT_LIB_FFTW 
        FREE_(pc->_ir_part_plans);
        FREE_(pc->_tmp_bufs_plans_fwd);
        FREE_(pc->_tmp_bufs_plans_bwd);
#endif  
        return err_MEM;
    }
    pc->M = M;
    pc->N = N;
    pc->d = d;
    return err_NONE;
}

void part_conv_destroy(part_conv_t *pc)
{
    size_t d;
    for (d = 0; d < pc->d; d++) {
#ifdef FFT_LIB_FFTW 
        fftw_destroy_plan(pc->_ir_part_plans[d]);
        FFTW_FREE_(pc->ir_parts[d]);
        fftw_destroy_plan(pc->_tmp_bufs_plans_fwd[d]);
        fftw_destroy_plan(pc->_tmp_bufs_plans_bwd[d]);
        FFTW_FREE_(pc->tmp_bufs[d]);
#endif  
    }
    FREE_(pc->ir_parts);
    FREE_(pc->tmp_bufs);
#ifdef FFT_LIB_FFTW 
    FREE_(pc->_ir_part_plans);
    FREE_(pc->_tmp_bufs_plans_fwd);
    FREE_(pc->_tmp_bufs_plans_bwd);
#endif  
    memset(pc,0,sizeof(part_conv_t));
}

void
part_conv_set_ir (part_conv_t *pc, f64_t *ir)
{
    size_t _d;
    for (_d = 0; _d < pc->d; _d++) {
#ifdef FFT_LIB_FFTW
        memset(pc->ir_parts[_d],0,(pc->_parts_sz/2+1)*sizeof(fftw_complex));
        memcpy(pc->ir_parts[_d],&ir[_d*pc->N/pc->d],pc->N/pc->d*sizeof(f64_t));
        /* divide by logical size N^2 because inverse tranform doesn't for both
         * IR transform and input transform */
        size_t n;
        for (n = 0; n < pc->N/pc->d; n++) {
            ((f64_t*)pc->ir_parts[_d])[n] /= (pc->_parts_sz*pc->_parts_sz);
        }
        fftw_execute(pc->_ir_part_plans[_d]);
#endif  
    }
}

/* x contains input and on return it will have been convolved with ir in pc.
 * x should have size pc->M + pc->N - 1 */
void
part_conv_do_conv (part_conv_t *pc, f64_t *x)
{
    size_t d;
    for (d = 0; d < pc->d; d++) {
        /* Do convolution of each part by multiplying in frequency domain */
#ifdef FFT_LIB_FFTW
        memset(pc->tmp_bufs[d],0,(pc->_parts_sz/2+1)*sizeof(fftw_complex));
        memcpy(pc->tmp_bufs[d],&x[d*pc->N/pc->d],pc->N/pc->d*sizeof(f64_t));
        fftw_execute(pc->_tmp_bufs_plans_fwd[d]);
        size_t n;
        for (n = 0; n < (pc->_parts_sz/2+1); n++) {
            pc->tmp_bufs[d][n] *= pc->ir_parts[d][n];
        }
        fftw_execute(pc->_tmp_bufs_plans_bwd[d]);
#endif  
    }
    /* Sum in all bufs */
    memset(x,0,sizeof(f64_t)*(pc->M+pc->N-1));
    size_t n;
    for (d = 0; d < (pc->d - 1); d++) {
        for (n = 0; n < (pc->M + pc->N/pc->d - 1); n++) {
            x[n] += ((f64_t*)pc->tmp_bufs[d])[n];
        }
        x += pc->N/pc->d;
    }
    /* Last buf might be shorter */
    for (n = 0; n < (pc->M + pc->N - 1 - pc->N/pc->d*(pc->d-1)); n++) {
        x[n] += ((f64_t*)pc->tmp_bufs[pc->d - 1])[n];
    }
}

#ifdef PART_CONV_TEST
//typedef struct param_set_t {
//    int M;
//    int N;
//    int d;
//    struct param_set_t *next;
//} param_set_t;

/* Test to see algorithm is correct */
void part_conv_correct_test(void)
{
    int M, N, d;
    M = (random() % 256) + 1;
    d = (int)pow(2.,(random() % 7) + 1);
    N = d * ((random() % 1000) + 1);
    f64_t *ir = CALLOC(f64_t,N);
    f64_t *out1 = CALLOC(f64_t,M+N-1);
    f64_t *out2 = CALLOC(f64_t,M+N-1);
    size_t n;
    for (n = 0; n < N; n++) {
        ir[n] = RAND_C64();
    }
    for (n = 0; n < M; n++) {
        out1[n] = out2[n] = RAND_C64();
    }
    part_conv_t pc;
    assert(part_conv_init(&pc,M,N,d) != err_MEM);
    part_conv_set_ir(&pc,ir);
    part_conv_do_conv(&pc,out1);
    part_conv_do_conv(&pc,out2);
    for (n = 0; n < (N+M-1); n++) {
        assert(abs(out1[n]-out2[n]) < 1e-6);
    }
    part_conv_destroy(&pc);
    free(ir);
    free(out1);
    free(out2);
}

#define N_CONVS 1 

int
main (void) {
    srandom(time(NULL));
    part_conv_correct_test();
    int M, N, d;
    while (fscanf(stdin,"%d %d %d\n",&M,&N,&d) == 3) {
        fprintf(stderr,"M=%d N=%d d=%d\n",M,N,d);
        /* build and do convolution */
        part_conv_t pc;
        assert(part_conv_init(&pc,M,N,d) != err_MEM);
        f64_t *tmp_buf = CALLOC(f64_t,M+N-1);
        f64_t *tmp_ir  = CALLOC(f64_t,N);
        size_t n;
        for (n = 0; n < M; n++) {
            tmp_buf[n] = RAND_C64();
        }
        for (n = 0; n < N; n++) {
            tmp_ir[n] = RAND_C64();
        }
        part_conv_set_ir(&pc,tmp_ir);
        /* start profiling */
        CALLGRIND_START_INSTRUMENTATION;
        for (n = 0; n < N_CONVS; n++) {
            part_conv_do_conv(&pc,tmp_buf);
        }
        CALLGRIND_STOP_INSTRUMENTATION;
        CALLGRIND_DUMP_STATS;
        part_conv_destroy(&pc);
        free(tmp_buf);
        free(tmp_ir);
    }
    return 0;
}
#endif  



