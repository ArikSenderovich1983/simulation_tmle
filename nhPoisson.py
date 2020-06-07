from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
         bytes, dict, int, list, object, range, str,
         ascii, chr, hex, input, next, oct, open,
         pow, round, super,
         filter, map, zip)

import numpy as np
from tick.base import TimeFunction
import matplotlib as plt
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson
from tick.survival import CoxRegression

#import NeuroTools
#from NeuroTools import stgen


from numpy import *
plt.use('tkagg')
def fit_Cox(features, T, E):
    cox_m = CoxRegression(verbose = True)
    cox_m.fit(features, T, E)
    return
def simulate_lognorm(arrivals, mu, sigma):

    #mean lognorm = 28.75
    #stdev = 37.47
    intervals = {}
    obs = []
    for j,a in enumerate(arrivals):
        #simulate a arrivals per part
        #sample = list(np.random.lognormal(mean = mu, sigma = sigma, size = a))
        sample = list(exp(np.random.normal( mu,  sigma, size = a)))

        obs.extend(sample)
        intervals[j] = sample
    return obs, intervals
def simulate_NHP(run_time, T, Y, dt_, track):
    #run_time = 30

    #T = np.arange((run_time * 0.9) * 5, dtype=float) / 5
    #Y = np.maximum(
    #    15 * np.sin(T) * (np.divide(np.ones_like(T),
    #                                np.sqrt(T + 1) + 0.1 * T)), 0.001)

    tf = TimeFunction((T, Y), dt=dt_)

    # We define a 1 dimensional inhomogeneous Poisson process with the
    # intensity function seen above
    in_poi = SimuInhomogeneousPoisson([tf], end_time=run_time, verbose=True, seed = 3)#, max_jumps=sum(Y))

    # We activate intensity tracking and launch simulation
    in_poi.track_intensity(track)
    in_poi.threshold_negative_intensity(allow=True)

    in_poi.simulate()

    # We plot the resulting inhomogeneous Poisson process with its
    # intensity and its ticks over time
    plot_point_process(in_poi)
    return list(in_poi.tracked_intensity[0]), list(in_poi.intensity_tracked_times)
def return_intensity(t, T, Y, last_w):
    for w in range(last_w, len(T)-1):
        if t>=T[w] and t<T[w+1]:
            return Y[w], w
def sim_NHP_Thinning(run_time,T, Y, time_windows):
    lambda_u = max(Y)
    arrival_times = []
    count_ = 0
    t_=0
    last_w = 0
    while True:
        u_1 = np.random.uniform(0,1)

        t_ = t_-(np.log(u_1)/lambda_u) #*time_windows
        if t_>run_time:
            break
        u_2 = np.random.uniform(0,1)
        lambda_t, last_w = return_intensity(t_,T,Y, last_w)
        if u_2<=lambda_t/lambda_u:
            arrival_times.append(t_)
            count_+=1
    return arrival_times
def inverse_CDF(ecdf, u):
    for x in ecdf.x:
        if ecdf(x)>=u:
            return x
def sample_empirical(ecdf):
    u_1 = np.random.uniform(0, 1)
    return inverse_CDF(ecdf, u_1)




#run_time = 30
#T = np.arange((run_time * 0.9) * 5, dtype=float) / 5
##Y = np.maximum(
#        15 * np.sin(T) * (np.divide(np.ones_like(T),
#                                    np.sqrt(T + 1) + 0.1 * T)), 0.001)

#simulate_NHP(run_time, T, Y)