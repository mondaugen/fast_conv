/* Routines for partition convolutions */

#include <stdio.h> 

#ifdef FFT_LIB_FFTW

#include <complex.h>
#include <fftw3.h> 

#define FFTW_MALLOC(n) (fftw_complex*)fftw_malloc((n)*sizeof(fftw_complex))
#define FFTW_FREE_(x) if (x) { fftw_free(x); x = NULL; } 

#endif  

#define FREE_(x)      if (x) { free(x); x = NULL; } 
#define CALLOC(t,n) (t*)calloc(sizeof(t),n) 
#define CONT_IF_NULL(x,flg) if (!x) { flg = 1; continue }
#define BRK_IF_NULL(x,flg,cond) if (!x) { cond; flg = 1; continue }

static inline void *MCHK (void *x)
{
    if (!x) {
        fprintf("Out of memory.");
        abort();
    }
    return x;
}

typedef enum err_t {
    err_NONE,
    err_EINVAL,
    err_MEM,
} err_t

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
                                                              pc->ir_parts[_d],
                                                              pc->ir_parts[_d],
                                                              FFTW_ESTIMATE);
                pc->tmp_bufs[_d] = FFTW_MALLOC(pc->_parts_sz/2+1);
                pc->_tmp_bufs_plans_fwd[_d] = fftw_plan_dft_r2c_1d(pc->_parts_sz,
                                                              pc->tmp_bufs[_d],
                                                              pc->tmp_bufs[_d],
                                                              FFTW_ESTIMATE);
                pc->_tmp_bufs_plans_bwd[_d] = fftw_plan_dft_c2r_1d(pc->_parts_sz,
                                                              pc->tmp_bufs[_d],
                                                              pc->tmp_bufs[_d],
                                                              FFTW_ESTIMATE);
                BRK_IF_NULL(pc->tmp_bufs[_d],_d=d);
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
    return err_NONE;
}

void
part_conv_set_ir (part_conv_t *pc, f64_t *ir)
{
    size_t _d;
    for (_d = 0; _d < pc->d; _d++) {
#ifdef FFT_LIB_FFTW
        memset(pc->ir_parts[_d],0,(pc->_parts_sz/2+1)*sizeof(fftw_complex));
        memcpy(pc->ir_parts[_d],&ir[_d*pc->N/pc->d],pc->N/pc->d);
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
        memset(pc->tmp_bufs[_d],0,(pc->_parts_sz/2+1)*sizeof(fftw_complex));
        memcpy(pc->tmp_bufs[_d],&ir[_d*pc->N/pc->d],pc->N/pc->d);
        fftw_execute(pc->_tmp_bufs_plans_fwd[d]);
        size_t n;
        for (n = 0; n < (pc->_parts_sz/2+1); n++) {
            pc->tmp_bufs[d] *= pc->ir_parts[d];
        }
        fftw_execute(pc->_tmp_bufs_plans_bwd[d]);
#endif  
    }
    /* Sum in all bufs */
    memset(x,0,sizeof(f64_t)*(pc->M+pc->N-1));
    for (d = 0; d < (pc->d - 1); d++) {
        size_t n;
        for (n = 0; n < (pc->M + pc->N/pd->d - 1); n++) {
            x[n] += pc->tmp_bufs[d][n];
        }
        x += pc->N/pc->d;
    }
    /* Last buf might be shorter */
    for (n = 0; n < (pc->M + pc->N - 1 - pc->N/pc->d*(d-1)); n++) {
        x[n] += pc->tmp_bufs[pc->d - 1][n];
    }
}
