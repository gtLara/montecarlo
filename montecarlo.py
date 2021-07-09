import numpy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Rotinas visuais

def see_dist(samples, title=None):
    plt.hist(samples, bins=100, density=True)
    plt.title(title)
    plt.show()

# Rotinas numericas

def infer(Rs, delta_Rd, delta_Rts, rc, r, delta_Rtx):

    Rx = (Rs + delta_Rd + delta_Rts) * rc * r - delta_Rtx

    return Rx

def get_Rs(mean=10000.053, std=0.0025):
    Rs = numpy.random.normal(mean, std) # gaussiana centrada em 10k com dp .0025
    return Rs

def get_delta_Rd(lower_lim=0.001, upper_lim=0.003):
    delta_Rs = numpy.random.uniform(lower_lim, upper_lim) # uniforme distribuida entre 10 e 30 mOhms
    return delta_Rs

#TODO: conferir delta_Rts

def get_delta_Rts(lower_lim=9999.99725, upper_lim=10000.00275):
    delta_Rts = numpy.random.uniform(lower_lim, upper_lim) # uniforme distribuida entre 9997.25 e 10002.75 Ohms
    return delta_Rts

def get_rc(lower_lim=0.999999, middle_value=1, upper_value=1.000001):
    rc = numpy.random.triangular(lower_lim, middle_value, upper_value, size=None)
    return rc

def get_delta_Rtx(r_mean=None, r_std=None):
    return 0

def get_r(mean=0, std=0):
    r = numpy.random.normal(mean, std)
    return r

def get_Rx(r_mean, r_std):
    Rx = infer(get_Rs(), get_delta_Rd(), get_delta_Rts(),
                get_rc(), get_r(r_mean, r_std), get_delta_Rtx())

    return Rx

# Definição de parâmetros r

r_samples = [1.0000104, 1.0000107, 1.0000106, 1.0000103, 1.0000105]
r_mean = numpy.mean(r_samples)
r_std = numpy.std(r_samples)/np.sqrt(len(r_samples))

M = 5000000 # numero de iteracoes

rtx_samples = [get_Rx(r_mean, r_std) for n in range(10000)]
