import pandas as pd
import numpy as np
import mpmath as mp
import sympy as sym
from sympy import pi,exp
from sympy.abc import a, t, s, x, y
from scipy.stats import bernoulli, expon
from zepid.causal.gformula import TimeFixedGFormula
from zepid import load_sample_data, spline
from zepid.causal.doublyrobust import TMLE
from zepid.causal.doublyrobust import StochasticTMLE

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



BATCH_SIZE = 1#28
EPOCHS = 10

#data = sm.datasets.scotland.load(as_pandas=False)
#print(data.exog)
#data_exog = sm.add_constant(data.exog)
#print(data_exog)
df = pd.read_csv("intervention_0.csv")
df = df.loc[0:1000,:]
df.reset_index(inplace=True, drop=True)
df['T'] = 0.0
df['D'] = 0.0
df['D_Lag'] = 0.0
df['S_Lag'] = 0.0
df['A_star'] = 0
p_1 = 0.7
new_si = []
new_si_lag = []


for i in range(len(df)):
    df.at[i,'D'] = df.at[i,'elapsed']-df.at[i,'S']
    if i==0:
        df.at[i,'T'] = df.at[i,'arrival_time']
        #df.at[i,'D_Lag'] = 0.0
        #df.at[i,'S_Lag'] = 0.0

    else:
        df.at[i,'T']= df.at[i,'arrival_time'] - df.at[i-1,'arrival_time']
        df.at[i,'D_Lag'] = df.at[i-1,'D']
        df.at[i,'S_Lag'] = df.at[i-1,'S']

    df.at[i,'A_star'] = bernoulli.rvs(p_1, size=1)

c_i = list(df['arrival_time'])
a_i = np.array(df['A'])
s_i = np.array(df['S'])
s_lag = np.array(df['S_Lag'])
#s_lag.insert(0,0.0)

#df.to_csv('forTMLE.csv', header = True, index =False)


y_i = np.array(df['elapsed'])
d_i = np.array(df['D']) #[y_i[i]-s_i[i] for i in range(len(s_i))]
print(len(d_i))
d_lag = np.array(df['D_Lag']) #[y_i[i-1]-s_i[i-1] for i in range(1,len(s_i))]

#d_lag.append(0)

#d_lag.insert(0,0.0)

print("Mean LOS: "+str(np.mean(y_i)))
c_i.insert(0,0.0)
t_i = np.array(df['T'])
new_ai = list(np.array(df['A_star']))

print(len(c_i), len(t_i), len(a_i), len(s_i))
method = 'TMLE' #'OoBox' # 'ML'

if method=='parametric':

    l, p, m1, m2 = sym.symbols('l,p,m1,m2',positive=True)

    L1=p**a*(1-p)**(1-a)
    J1=np.prod([L1.subs(a,i) for i in a_i])
    print(J1)

    L2=l*sym.exp(-l*t)
    J2=np.prod([L2.subs(t,i) for i in t_i])
    print(J2)

    L3=sym.Add((1-a)*m1*sym.exp(-m1*s),a*m2*sym.exp(-m2*s))
    J3=np.prod([L3.subs({a:i, s:s_i[j]}) for j,i in enumerate(a_i)])
    print(J3)
    print(sym.expand_log(sym.log(J3)))
    logJ = sym.expand_log(sym.log(J1*J2*J3))
    print(logJ)

    sol_p=float(sym.solve(sym.diff(logJ,p),p)[0])
    sol_l=float(sym.solve(sym.diff(logJ,l),l)[0])
    sol_m1=float(sym.solve(sym.diff(logJ,m1),m1)[0])
    sol_m2=float(sym.solve(sym.diff(logJ,m2),m2)[0])

    print(sol_p, sol_l, sol_m1, sol_m2)

    #now, I need to generate new S given new p':



    #ti are the same
    for a_i in new_ai:
        if a_i==1:
            new_si.append(expon.rvs(scale = 1/sol_m2))
        else:
            new_si.append(expon.rvs(scale = 1/sol_m1))
    print("New S_i: ", np.mean(new_si))

    #L1_cur = 1.0
    #L2_cur = 1.0
    #L3_cur = 1.0
    #new_yi = []

    #for i, a_new in enumerate(new_ai):
        #print("New ai:",new_ai)
    #    L1_cur = L1_cur*L1.subs({x:a_new, p:p_1})
    #    L2_cur = L2_cur * L2.subs({t:t_i[i], l:sol_l})
    #    L3_cur = L3_cur * L3.subs({s:new_si[i], m1:sol_m1, m2:sol_m2, x:a_new})
        #print(L1_cur, L2_cur, L3_cur)
    #    new_yi.append(y_i[i]*L1_cur*L2_cur*L3_cur)
    #print("LOS after change (13): " + str(np.sum(new_yi)))

    new_di = []
    cur_di = 0
    new_yi = []
    new_si_lag = [new_si[i] - new_si[i - 1] for i in range(1, len(s_i))]

    new_si_lag.insert(0, 0)

    for j,s in enumerate(new_si):
        temp_di = max(0,cur_di+s-t_i[j])
        new_di.append(temp_di)
        new_yi.append(temp_di+s)#new_si[j])
        cur_di = temp_di
    mean_yi = np.mean(new_yi)
    print("LOS after change: "+str(mean_yi))

    var_y = pow(sum([pow(y-mean_yi,2) for y in new_yi])/(len(new_yi)-1),0.5)
    print("STDEV of LOS after change: "+str(var_y))

    X_train = np.concatenate((np.array(d_lag).reshape(-1, 1), s_lag.reshape(-1, 1), t_i.reshape(-1, 1)), axis=1)
    y_train = np.array(d_i)  # .reshape(-1,1)

    model = tfile.Perceptron(3, 1)
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

            if (i % 10 == 0):
                print("Epoch {} - loss: {}".format(epoch, loss.data))

    print(list(model.parameters()))

    cur_di = 0
    new_yi = []
    new_di = []
    for i in range(len(new_si)):
        pred_val = float(model(Variable(torch.Tensor([[[cur_di, new_si[i], t_i[i]]]]))).data[0][0][0])
        new_di.append(pred_val)
        new_yi.append(new_si[i] + pred_val)
        cur_di = pred_val

    print(np.mean(np.array(new_yi)))

elif method == 'OoBox':
    g = TimeFixedGFormula(df, exposure='A', outcome='S', outcome_type='normal')
    g.outcome_model(model='A')

    g.fit(treatment="g['A'] == 1")
    effect_old = g.marginal_outcome
    #g.run_diagnostics()
    g.fit_stochastic(p=p_1, seed=1000191) #treatment="g['A_star'] == 1")
    #g.run_diagnostics()

    effect_new = g.marginal_outcome

    print("old effect: "+str(effect_old))
    print("new effect: "+str(effect_new))

elif method =='TMLE':
    df_tmle = df[['C','A','T','S_Lag','D_Lag','D', 'S']]
    df_tmle.reset_index(drop=True, inplace=True)
    #implementing TMLE with supylearner
    print("TMLE start:")
    M = MLPRegressor(random_state=1, max_iter=500)
    tml = TMLE(df_tmle, exposure='A', outcome='S')
    tml.exposure_model('1')
    tml.outcome_model('A',custom_model = M)
    tml.run_diagnostics()
    tml.fit()
    tml.summary()

#elif method == 'ML':
    #data_exog = sm.add_constant(a_i.reshape(-1,1))
    #data_exog = sm.add_constant(np.array(a_i).reshape(-1,1))
    #print(data_exog)
    #gauss_log = sm.GLM(s_i, data_exog, family=sm.families.Gaussian(sm.families.links.log))
    #gauss_log_results = gauss_log.fit()
    #print(gauss_log_results.summary())

    #new_si = gauss_log_results.predict(sm.add_constant(np.array(new_ai).reshape(-1,1))) #, exposure = new_ai)))


















    #model = Sequential()
    #model.add(Dense(1, activation="relu", input_dim=2, name="layer" + str(0)))

    #for i in range(1,len(s_i)):
    #    model.add(Dense(1, activation = "relu", input_dim=1, name = "layer"+str(i)))
    #    print(i)
    #model.compile(optimizer='sgd',
    #                loss='mse',
    #        metrics=[keras.metrics.MeanSquaredError()])
    #model.fit(X_train, y_train,
    #          batch_size=BATCH_SIZE,
    #          epochs=EPOCHS,
              #callbacks=[plot_losses],
    #          verbose=1)
    #print('predicted CF delay', np.array(model.predict(X_test)))
    #new_yi = np.array(model.predict(X_test)) + np.array(new_si)

    #old_yi = np.array(model.predict(X_train)) + np.array(s_i)

    #print('Old Mean Y_i: ' + str(np.mean(old_yi)))

    #print('New Mean Y_i: ' + str(np.mean(new_yi)))
    #y_test = np.array(d_i).reshape(-1,1)

    #need to fit GLM tomponent of the likelihood
    #I will use Keras (using NN for the nonparam)
    #First regression: S|A (log-linear)

    #sigma2, beta, beta0 = sym.symbols('sigma2, beta, beta0', positive=True)


    #log_s = np.array([np.log(s) for s in s_i])

    #L_lreg = pow((2*pi*sigma2),-0.5)*exp(-0.5/sigma2*pow(y-(beta*x+beta0),2))
    #print(L_lreg)
    #J1 = np.prod([L_lreg.subs({y:log_s[i],x:a_i[i]}) for i in range(len(log_s))])
    #print(J1)

    #logJ = sym.expand_log(sym.log(J1))
    #print(logJ)

    #sol_sig = float(sym.solve(sym.diff(logJ, sigma2), sigma2)[0])
    #sol_beta = float(sym.solve(sym.diff(logJ, beta), beta)[0])
    #sol_beta0 = float(sym.solve(sym.diff(logJ, beta0), beta0)[0])

    #print(sol_sig, sol_beta, sol_beta0)



    #print(a_i.reshape(-1,1))
    #reg = LinearRegression().fit(a_i.reshape(-1,1), log_s)
    #print(reg.score(a_i.reshape(-1,1),log_s))
    #print(reg.coef_)
    #print(reg.intercept_)

    #pred_s = [exp(a * reg.coef_[0] + reg.intercept_) for a in new_ai]
    #print("Predicted avg: ", np.mean(np.array(pred_s)))
    #print(np.mean(np.array(pred_s) - s_i))
    #print(np.mean(pow(np.array(pred_s) - s_i, 2)))

    #Second regression: D|S,A,T




    #    print('yay')
