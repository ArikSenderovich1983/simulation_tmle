from sklearn.model_selection import GridSearchCV
import sympy as sym
from sympy.abc import a, t, s, x, y
from lifelines import CoxPHFitter
import random as rd
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import torch_file as tfile
from torch_file import *
BATCH_SIZE = 1  # 28
EPOCHS = 10
def pytorch_learn_lindley(df, target_col, num_cols):
    # predicting D_i
    print(df.head())
    print(len(df))
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    p_train = 1#0.8

    # Train Test Split

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    train_indices = [i for i in range(0, int(len(X) * p_train))]
    test_indices = [i for i in range(int(len(X) * p_train), len(X))]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Scaling Data for ANN
    #sc_X = StandardScaler()
    #X_train = sc_X.fit_transform(X_train)
    #X_test = sc_X.transform(X_test)

    #Unknown Lindley - replaced with ReLU perceptron

    #dict_w = {"T_i": t_i, "S_lag": s_prev, "W_lag": w_prev, "W_target": w_i}  # , "W_lindley": w_lindley}
    d_lag = np.array(X_train.W_lag.values)
    s_lag = np.array(X_train.S_lag.values)
    t_i = np.array(X_train.T_i.values)
    d_i = np.array(y_train.values)
    X_train = np.concatenate((np.array(d_lag).reshape(-1, 1), s_lag.reshape(-1, 1), t_i.reshape(-1, 1)), axis=1)
    y_train = np.array(d_i)  # .reshape(-1,1)

    model = tfile.relu_Perceptron(3, 1)
    print(model)

    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.5)
    optimizer = optim.Adadelta(model.parameters())  # , momentum=0.5)

    # data = np.concatenate(X_train,y_train, axis = 1)

    for epoch in range(EPOCHS):
        for i, X in enumerate(X_train):
            X = Variable(torch.FloatTensor([X]), requires_grad=True)
            Y = Variable(torch.FloatTensor([y_train[i]]), requires_grad=False)

            optimizer.zero_grad()
            #X.grad.data.zero_()
            #Y.grad.data.zero_()

            outputs = model(X)
            # loss = criterion(outputs, Y)
            criterion = nn.MSELoss()
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
    print("NN parameters:s")
    print(list(model.parameters()))

    return model
def pytorch_predict_lindley(new_si, new_ti, model):
    new_di = []
    new_yi = []
    cur_di = 0

    for i in range(1,len(new_si)-1):


        pred_val = float(model(Variable(torch.Tensor([[[cur_di, new_si[i-1], new_ti[i]]]]))).data[0][0][0])

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

    mean_yi = np.mean(new_yi)
    print("LOS after change: " + str(mean_yi))
    var_y = pow(np.sum([pow(y - mean_yi, 2) for y in new_yi]) / (len(new_yi) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_y))

    mean_di = np.mean(new_di)
    print("Waiting after change: " + str(mean_di))
    var_d = pow(np.sum([pow(d - mean_di, 2) for d in new_di]) / (len(new_di) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_d))


    return new_yi, new_di, var_y, var_d



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
def mg1_speedup_semiparam_service_model(new_ai, a_i, s_i):
    ai_temp = []
    for a in a_i:
        if a==0:
            ai_temp.append(-1)
        else:
            ai_temp.append(1)
    data = {'a_i':ai_temp, 's_i':s_i, 'e_i':np.ones(len(s_i))}
    df = pd.DataFrame.from_dict(data)
    cph = CoxPHFitter()
    cph.fit(df, 's_i', 'e_i')#,strata=['a_i'])

    new_si = []
    new_ai_temp = []
    for a in new_ai:
        if a==0:
            new_ai_temp.append(-1)
        else:
            new_ai_temp.append(1)
    for a_new in new_ai_temp:

        log_U = np.log(rd.uniform(0,1))
        try:
            sample_ = inv_cum_hazard(cph.baseline_cumulative_hazard_,-log_U * np.exp(-cph.params_[0]*a_new))

            new_si.append(sample_)

        except ValueError:
            print('here')
            new_si.append(np.mean(s_i))
        #cph.baseline_cumulative_hazard_
        #sample_ = -log_U * np.exp(-cph.params_[0]*a_new) / lam_#inv_cum_hazard(cph.baseline_cumulative_hazard_,-log_U * np.exp(-cph.params_[0]*a_new))#cph.baseline_cumulative_hazard_



    print("New s_i nonparametric", np.mean(np.array(new_si)))
    #results = proportional_hazard_test(cph, df, time_transform='rank')
    #results.print_summary(decimals=3, model="untransformed variables")
    return new_si

def inv_cum_hazard(H, val):
    #print("Value:", val)
    for i,h in enumerate(H.values):
        #print(H.at[i,'baseline cumulative hazard'])
        if h[0]>=val:
            return H.index[i]
    raise ValueError   #H.index[len(H.values)-1]
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
    optimizer = optim.Adadelta(model.parameters())  # , momentum=0.5)

    # data = np.concatenate(X_train,y_train, axis = 1)

    for epoch in range(EPOCHS):
        for i, X in enumerate(X_train):
            X = Variable(torch.FloatTensor([X]), requires_grad=True)
            Y = Variable(torch.FloatTensor([y_train[i]]), requires_grad=False)

            optimizer.zero_grad()
            #X.grad.data.zero_()
            #Y.grad.data.zero_()

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
    #new_si.insert(0,0)
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

    mean_yi = np.mean(new_yi)
    print("LOS after change: " + str(mean_yi))
    var_y = pow(np.sum([pow(y - mean_yi, 2) for y in new_yi]) / (len(new_yi) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_y))

    mean_di = np.mean(new_di)
    print("Waiting after change: " + str(mean_di))
    var_d = pow(np.sum([pow(d - mean_di, 2) for d in new_di]) / (len(new_di) - 1), 0.5)
    print("STDEV of LOS after change: " + str(var_d))


    return new_yi, new_di, var_y, var_d
def mg1_cf_los_NN_model(d_lag, s_lag, t_i, d_i, new_si):
    # predicting D_i


    #Unknown Lindley - replaced with ReLU perceptron
    new_di = []
    new_yi = []


    X_train = np.concatenate((np.array(d_lag).reshape(-1, 1), s_lag.reshape(-1, 1), t_i.reshape(-1, 1)), axis=1)
    y_train = np.array(d_i)  # .reshape(-1,1)

    nn = MLPRegressor(max_iter=1000, activation='relu')
    parameter_space = {
        'hidden_layer_sizes': [(pow(2,i),j) for i in range(0,6) for j in [1, 10, 20]]}  #[(50, 50, 50), (50, 100, 50), (100,)],
        #'activation': ['tanh', 'relu', 'indentity']}
        #'max_iter': [500, 600, 700],
        #'learning_rate_init': [0.001, 0.01, 0.1],
        #'solver': ['sgd', 'adam'],
        #'alpha': [0.0001, 0.05],
        #'learning_rate': ['constant', 'adaptive']}
    #}
    #todo: add printout of best model

    clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=3, verbose=1)
    clf.fit(X_train, y_train)
    print("Best model is", clf.best_params_)
    cur_di = 0
    #print(len(t_i),len(new_si))
    #input('wait')
    for i in range(1,len(new_si)-1):


        pred_val = clf.predict(np.array([cur_di, new_si[i-1], t_i[i]]).reshape(1,-1))[0]

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
if False:
    for r in run_list:
        print('Run number '+str(r))
        for p_1 in p_list:
                if lam_==1.0:
                    lam_ = 1
                df_cf = pd.read_csv("datafiles\\intervention_data_"+str(lam_)+"_"+str(int(mu_2))+"_"+str(p_1)+".csv")

                real_los = np.mean(df_cf.loc[n_min:,'elapsed'].values)
                real_wait = np.mean(df_cf.loc[n_min:, 'elapsed']-df_cf.loc[n_min:,'S'])
                real_std_los = np.std(df_cf.loc[n_min:,'elapsed'].values)
                real_std_wait = np.std(df_cf.loc[n_min:, 'elapsed']-df_cf.loc[n_min:,'S'])

                new_ai, d_lag, s_lag, d_i, y_i = enrich_df(df, p_1)

                main_param_only(data, p_1, mu_2, r, p_speed, lam_, real_los,
                                real_wait, real_std_los, real_std_wait, sol_m1, sol_m2,
                                a_i, t_i, s_i, new_ai, d_lag, s_lag, d_i, n_min, steps = 30)


        df_dict = pd.DataFrame.from_dict(data)
        df_dict.to_csv('counterfactual_mm1_results_'+str(K)+'_'+str(n_rows)+'_runs.csv')