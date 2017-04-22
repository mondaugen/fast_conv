import numpy as np
import matplotlib.pyplot as plt
def next_pow_2(x):
    return np.power(2.,np.ceil(np.log(x)/np.log(2)))

def part_conv_comp_per_samp(M,N,d):
    """
    M length of input.
    N length of IR.
    d number of partitions.
    """
    return (( # fourier transform part
            (d+1)
            * next_pow_2(M+N/d-1)*np.log(next_pow_2(M+N/d-1))/np.log(2)
            # multiplication part
            + d*next_pow_2(M+N/d-1)
            # addition part
            + (d-1)*(M+N/d-1)
            # divide by number of input samples
            ) / M)

d=np.power(2.,np.arange(20))
M=64.
N=8200.
c=part_conv_comp_per_samp(M,N,d)
print 'min time at number of partitions %f', (d[np.argmin(c)],)
print 'each of size', (next_pow_2(M+N/d[np.argmin(c)]-1),)
plt.plot(np.log(d)/np.log(2),np.log(c)/np.log(2))
plt.show()
