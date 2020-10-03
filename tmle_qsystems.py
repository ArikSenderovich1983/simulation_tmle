import pandas as pd
import numpy as np
import mpmath as mp
import sympy as sym
from sympy import pi,exp
from sympy.abc import a, t, s, x, y
from scipy.stats import bernoulli, expon
from zepid.causal.gformula import TimeFixedGFormula
from zepid import load_sample_data, spline
from zepid.causal.doublyrobust import TMLE, StochasticTMLE
#from zepid.causal.doublyrobust import StochasticTMLE
from sklearn.linear_model import TweedieRegressor
import scipy
from deepSuperLearner import *
import supylearner as sl
from lifelines import CoxPHFitter
from pynverse import inversefunc
import random as rd
from KDEpy import FFTKDE

# Load dependencies
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.neural_network import MLPRegressor
#from keras import optimizers
#from keras import metrics
#import livelossplot
#plot_losses = livelossplot.PlotLossesKeras()
import torch_file as tfile
from torch_file import *
from matplotlib import pyplot
import pylab
from lifelines.statistics import proportional_hazard_test


from lifelines import WeibullAFTFitter




BATCH_SIZE = 1  # 28
EPOCHS = 10


def mg1_cf_los_nonparam_model(d_lag, s_lag, t_i, d_i, new_si):
    # predicting D_i


    #Unknown Lindley - replaced with ReLU perceptron
    new_di = []
    new_yi = []


    X_train = np.concatenate((np.array(d_lag).reshape(-1, 1), s_lag.reshape(-1, 1), t_i.reshape(-1, 1)), axis=1)
    y_train = np.array(d_i)  # .reshape(-1,1)

    model = tfile.relu_Perceptron(3, 1)
    print(model)

    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.5)
    optimizer = optim.Adagrad(model.parameters(), lr=0.1, lr_decay=0.0)  # , momentum=0.5)

    # data = np.concatenate(X_train,y_train, axis = 1)

    for epoch in range(EPOCHS):
        for i, X in enumerate(X_train):
            X = Variable(torch.FloatTensor([X]), requires_grad=True)
            Y = Variable(torch.FloatTensor([y_train[i]]), requires_grad=False)

            optimizer.zero_grad()
            outputs = model(X)
            # loss = criterion(outputs, Y)
            criterion = nn.MSELoss()
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            #if (i % 10000 == 0):
            #    print("Epoch {} - loss: {}".format(epoch, loss.data))
    print("NN parameters:s")
    print(list(model.parameters()))
    new_si.insert(0,0)
    lindley = []
    cur_di = 0
    #print(len(t_i),len(new_si))
    #input('wait')
    for i in range(1,len(new_si)-1):


        pred_val = float(model(Variable(torch.Tensor([[[cur_di, new_si[i-1], t_i[i]]]]))).data[0][0][0])

        #lindley.append(new_si[i-1]+max(0,cur_di+new_si[i-1]-t_i[i]))
        #if i>100 and i<200:
        #    print("Input:", cur_di,new_si[i-1], t_i[i])
        #    print("Prediction:", pred_val)
        #    print("Lindley:", max(0,cur_di+new_si[i-1]-t_i[i]))
        new_di.append(pred_val)
        new_yi.append(new_si[i-1] + pred_val)
        cur_di = pred_val
    print("Average LOS:")
    print(np.mean(np.array(new_yi)))
    print("Average Wait:")
    print(np.mean(np.array(new_di)))
    #print(np.mean(np.array(lindley)))
    return new_yi, new_di


def mg1_cf_los_param_model(t_i, new_si):

    #known Lindley

    new_di = []
    new_yi = []

    cur_di = 0
    #new_si_lag = [new_si[i] - new_si[i - 1] for i in range(1, len(new_si))]
    #new_si_lag.insert(0, 0)
    new_si.insert(0,0)
    for j in range(1,len(new_si)-1):
        temp_di = max(0, cur_di + new_si[j-1] - t_i[j])
        new_di.append(temp_di)
        new_yi.append(temp_di + new_si[j-1])  # new_si[j])
        cur_di = temp_di

    mean_yi = np.mean(new_yi)
    print("LOS after change: " + str(mean_yi))
    var_y = pow(sum([pow(y - mean_yi, 2) for y in new_yi]) / (len(new_yi) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_y))


    mean_di = np.mean(new_di)
    print("Waiting after change: " + str(mean_di))
    var_d = pow(sum([pow(d - mean_di, 2) for d in new_di]) / (len(new_di) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_d))


    return new_yi, new_di, var_y, var_d

def inv_cum_hazard(H, val):
    #print("Value:", val)
    for i,h in enumerate(H.values):
        #print(H.at[i,'baseline cumulative hazard'])
        if h[0]>=val:
            return H.index[i]

def mg1_speedup_semiparam_service_model(new_ai, a_i, s_i):
    ai_temp = []
    for a in a_i:
        if a==0:
            ai_temp.append(-1)
        else:
            ai_temp.append(1)
    data = {'a_i':ai_temp, 's_i':s_i, 'e_i':np.ones(len(s_i))}
    #todo: fix this because it produces negative values
    df = pd.DataFrame.from_dict(data)
    cph = CoxPHFitter()
    cph.fit(df, 's_i', 'e_i')#,strata=['a_i'])
    #print(cph.params_)
    #cph.print_summary()
    #cph.baseline_cumulative_hazard_['baseline cumulative hazard'].plot()
    #pyplot.show()
    #pyplot.scatter(x = cph.baseline_cumulative_hazard_.index, y = cph.baseline_cumulative_hazard_['baseline cumulative hazard'])
    #pyplot.show()
    #print("baseline hazard: ", cph.baseline_hazard_)
    #lp = LinearRegression(fit_intercept=False).fit(np.array(cph.baseline_cumulative_hazard_.index).reshape(-1, 1),
     #                                              cph.baseline_cumulative_hazard_['baseline cumulative hazard'])
    #print(lp.coef_)
    #mean_a = np.mean(a_i)
    #input("Wait")
    new_si = []
    new_ai_temp = []
    for a in new_ai:
        if a==0:
            new_ai_temp.append(-1)
        else:
            new_ai_temp.append(1)
    for a_new in new_ai_temp:

        log_U = np.log(rd.uniform(0,1))
        sample_ = inv_cum_hazard(cph.baseline_cumulative_hazard_,-log_U * np.exp(-cph.params_[0]*a_new))

        #cph.baseline_cumulative_hazard_
        #sample_ = -log_U * np.exp(-cph.params_[0]*a_new) / lam_#inv_cum_hazard(cph.baseline_cumulative_hazard_,-log_U * np.exp(-cph.params_[0]*a_new))#cph.baseline_cumulative_hazard_

        if sample_<0:
            print(sample_)

        new_si.append(sample_)

    print("New s_i nonparametric", np.mean(np.array(new_si)))
    #results = proportional_hazard_test(cph, df, time_transform='rank')
    #results.print_summary(decimals=3, model="untransformed variables")
    return new_si



def mg1_speedup_param_service_model(new_ai, sol_m1, sol_m2):
    # ti are the same
    new_si = []
    for a_new in new_ai:
        if a_new == 1:
            new_si.append(expon.rvs(scale=1 / sol_m2))
        else:
            new_si.append(expon.rvs(scale=1 / sol_m1))
    print("New average S_i: ", np.mean(new_si))

    return new_si

def MLE_(a_i,t_i,s_i):
    l, p, m1, m2 = sym.symbols('l,p,m1,m2', positive=True)

    L1 = p ** a * (1 - p) ** (1 - a)
    J1 = np.prod([L1.subs(a, i) for i in a_i])
    print(J1)

    L2 = l * sym.exp(-l * t)
    J2 = np.prod([L2.subs(t, i) for i in t_i])
    print(J2)

    L3 = sym.Add((1 - a) * m1 * sym.exp(-m1 * s), a * m2 * sym.exp(-m2 * s))
    J3 = np.prod([L3.subs({a: i, s: s_i[j]}) for j, i in enumerate(a_i)])
    print(J3)
    print(sym.expand_log(sym.log(J3)))
    logJ = sym.expand_log(sym.log(J1 * J2 * J3))
    print(logJ)

    sol_p = float(sym.solve(sym.diff(logJ, p), p)[0])
    sol_l = float(sym.solve(sym.diff(logJ, l), l)[0])
    sol_m1 = float(sym.solve(sym.diff(logJ, m1), m1)[0])
    sol_m2 = float(sym.solve(sym.diff(logJ, m2), m2)[0])

    print(sol_p, sol_l, sol_m1, sol_m2)
    return sol_p, sol_l, sol_m1, sol_m2

def enrich_df(df, p_1):
    df['T'] = 0.0
    df['D'] = 0.0
    df['D_Lag'] = 0.0
    df['S_Lag'] = 0.0
    df['A_star'] = 0
    df['A_Lag'] = 0
    for i in range(len(df)):
        df.at[i, 'D'] = df.at[i, 'elapsed'] - df.at[i, 'S']
        if i == 0:
            df.at[i, 'T'] = df.at[i, 'arrival_time']
            # df.at[i,'D_Lag'] = 0.0
            # df.at[i,'S_Lag'] = 0.0

        else:
            df.at[i, 'T'] = df.at[i, 'arrival_time'] - df.at[i - 1, 'arrival_time']
            df.at[i, 'D_Lag'] = df.at[i - 1, 'D']
            df.at[i, 'S_Lag'] = df.at[i - 1, 'S']
            df.at[i, 'A_Lag'] = df.at[i - 1, 'A']
        df.at[i, 'A_star'] = bernoulli.rvs(p_1, size=1)

    c_i = list(df['arrival_time'])
    a_i = np.array(df['A'])
    s_i = np.array(df['S'])
    s_lag = np.array(df['S_Lag'])
    # s_lag.insert(0,0.0)

    # df.to_csv('forTMLE_MC.csv', header = True, index =False)

    y_i = np.array(df['elapsed'])
    d_i = np.array(df['D'])  # [y_i[i]-s_i[i] for i in range(len(s_i))]
    print(len(d_i))
    d_lag = np.array(df['D_Lag'])  # [y_i[i-1]-s_i[i-1] for i in range(1,len(s_i))]

    # d_lag.append(0)

    # d_lag.insert(0,0.0)

    print("Mean LOS: " + str(np.mean(y_i)))
    print("Mean Wait: " + str(np.mean(d_i)))

    c_i.insert(0, 0.0)
    t_i = np.array(df['T'])
    new_ai = list(np.array(df['A_star']))

    print(len(c_i), len(t_i), len(a_i), len(s_i))

    df = df[['D', 'S', 'A', 'A_Lag', 'A_star', 'T', 'D_Lag', 'S_Lag']]  # , 'C']]
    df.reset_index(inplace=True, drop=True)
    return a_i, t_i, s_i, new_ai, d_lag, s_lag, d_i


def main(method, filename, p_1):
    # df = df.loc[0:100000,:]
    # df.reset_index(inplace=True, drop=True)
    df = pd.read_csv(filename)

    #p_1 = 0.7
    print("P_star is ", p_1)

    a_i, t_i, s_i, new_ai, d_lag, s_lag, d_i = enrich_df(df,p_1)
    #'OoBox' # 'ML'


    if method=='ParamKnownL':

        sol_p, sol_l, sol_m1, sol_m2 = MLE_(a_i,t_i,s_i)
        #now, I need to generate new S given new p':
        new_si = mg1_speedup_param_service_model(new_ai, sol_m1, sol_m2)
        new_yi, new_di, var_y, var_d = mg1_cf_los_param_model(t_i, new_si)

    elif method=='ParamUnknownL':

        #mg1_cf_los_nonparam_model(d_lag, s_lag, t_i, d_i, new_si)

        #new_si = mg1_speedup_nonparam_service_model(new_ai, s_i, a_i)
        #print("new si:", new_si[0:10])
        #print(len(new_si), len(new_ai))
        #input("Wait")
        #mg1_cf_los_param_model(t_i, new_si)
        sol_p, sol_l, sol_m1, sol_m2 = MLE_(a_i,t_i,s_i)

        new_si = mg1_speedup_param_service_model(new_ai, sol_m1, sol_m2)
        new_yi, new_di = mg1_cf_los_nonparam_model(d_lag, s_lag, t_i, d_i, new_si)

    elif method == 'SemiParamKnownL':

        # mg1_cf_los_nonparam_model(d_lag, s_lag, t_i, d_i, new_si)

        # new_si = mg1_speedup_nonparam_service_model(new_ai, s_i, a_i)
        # print("new si:", new_si[0:10])
        # print(len(new_si), len(new_ai))
        # input("Wait")
        # mg1_cf_los_param_model(t_i, new_si)
        #sol_p, sol_l, sol_m1, sol_m2 = MLE_(a_i, t_i, s_i)

        new_si = mg1_speedup_semiparam_service_model(new_ai, a_i, s_i)
        new_yi, new_di, var_y, var_d = mg1_cf_los_param_model(t_i, new_si)

    elif method == 'SemiParamUnknownL':
        new_si = mg1_speedup_semiparam_service_model(new_ai, a_i, s_i)
        new_yi, new_di = mg1_cf_los_nonparam_model(d_lag, s_lag, t_i, d_i, new_si)
    return new_yi, new_di


method = "ParamKnownL"
filename = "intervention_mu_2.csv"
data = {}
K=1
run_list = list(range(0,K))
#p_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
p_list = [0.7]

avg_yi = {}
avg_di = {}
for r in run_list:
    for p_1 in p_list:
        new_yi, new_di = main(method, filename, p_1)
        data["Waiting_Run_"+str(r)+"Speedup_"+str(p_1)] = new_di
        data["LOS_Run_"+str(r)+"Speedup_"+str(p_1)] = new_yi
        if "p_1" not in avg_di:
            avg_di["p_1"] = np.array(new_di)
            avg_yi["p_1"] = np.array(new_yi)

        else:
            avg_di["p_1"] += (avg_di["p_1"] - np.array(new_di))/(r+1)
            avg_yi["p_1"] += (avg_yi["p_1"] - np.array(new_yi))/(r+1)


df_waiting = pd.DataFrame.from_dict(avg_di)

df_waiting.to_csv('waiting_output_'+str(method)+'.csv')

df_los = pd.DataFrame.from_dict(avg_yi)
df_los.to_csv('los_output_'+str(method)+'.csv')

