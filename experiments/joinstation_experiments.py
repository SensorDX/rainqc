
from model.hurdle_regression import MixLinearModel
import pandas as pd
from util.evaluation import roc_metric
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, HuberRegressor



def asmatrix(x):
    return x.as_matrix().reshape(-1,1)

def nearby_stations(site_code, k=10, radius=500):
    stations = pd.read_csv("../localdatasource/nearest_stations.csv")  # Pre-computed value.
    k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]

    k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)['to'][0:k]

    #available_stations = LocalDataSource.__available_station(k_nearest, k)
    return k_nearest.tolist() #available_stations


def random_rainy_days(faults=5):
    w = np.random.randn(100) + 0.6
    r = (w>0).astype(int)
    print r
    x = r.copy()
    rainy_day = np.where(r==1)[0]
    d = np.random.choice(rainy_day ,faults)
    print "injected index",d
    indx = []
    done = False
    for i, ix in enumerate(d):
        mxdays = 5
        x[ix] = 9
        indx.append(ix)
        j = ix +1
        while((mxdays>0) and x[j]==1):
            x[j] = 9
            mxdays -=1
            indx.append(j)
            j +=1
            if len(indx)>=faults:
                done = True
                break

        if done:
            #len(indx)>=faults:
            break

    print "after injection"
    print indx
    print x

def synthetic_fault(observations, plot=True, alpha=0.01, f_type='Flat'):

    num_faults = int(np.ceil(alpha * len(observations)))
    # insert flatline
    print "Total anomalies", num_faults
    threshold = 10.0
    dt = observations.copy()
    rainy_day = np.where(dt > threshold)[0]
    d = np.random.choice(rainy_day, num_faults)
    print "injected index", d
    injected_indx = []
    done = False
    for i, ix in enumerate(d):
        if ix in injected_indx:
            continue
        mxdays = 5
        dt[ix] = 0.0
        injected_indx.append(ix)
        j = ix + 1
        while ((mxdays > 0) and dt[j]>threshold):
            dt[j] = 0.0
            mxdays -= 1
            injected_indx.append(j)
            j += 1
            if len(injected_indx) >= num_faults:
                done = True
                break

        if done:
            # len(indx)>=faults:
            break
    lbl = np.zeros([dt.shape[0]])
    lbl[injected_indx] = 1.0
    if plot:
        #plt.plot(dt, '.r', label="faults")

        plt.plot(observations, '.b', label="Observations")
        plt.plot(injected_indx, dt[injected_indx], '.r', label="faults")

        plt.legend(loc='best')
        #plt.show()
    #print injected_indx
    return dt, lbl



def synthetic_fault_old(observations, plot=False, alpha=0.01, f_type='Flat'):
    num_faults = int(np.ceil(alpha*len(observations)))
    # insert flatline
    print "Total anomalies", num_faults
    ix = np.argsort(observations,axis=0)
    dt = observations.copy()
    if f_type == 'Flat':
        rainy_days = np.where(observations>0.05)
        abnormal_days = ix[-(num_faults+2):-2]
        dt[abnormal_days] = 0.0
    elif f_type =='Spike':
        abnormal_days = ix[:num_faults]
        dt[abnormal_days] = 20.0
    else:
        abnormal_days_spike = ix[:(num_faults/2)]
        dt[abnormal_days_spike] = 20.0
        abnormal_days_flat = ix[-(num_faults/2 + 2):-2]
        dt[abnormal_days_flat] = 0.0
        abnormal_days = np.concatenate([abnormal_days_spike, abnormal_days_flat])

    #abnormal_report = range(200, 210)
    #rainy_days = range(107, 117)
    #dt[abnormal_report] = 20.0
    #dt[rainy_days] = 0.0
    lbl = np.zeros([dt.shape[0]])
    lbl[abnormal_days] = 1.0
    if plot:
        plt.plot(dt, '.r', label="faults")
        plt.plot(observations, '.b', label="Observations")
        plt.legend(loc='best')
        #plt.show()
    return dt, lbl

def plot_synthetic(dt, y):
    plt.plot(dt, '.r', label='inserted faults')
    plt.plot(y, '.b', label='ground truth')
    plt.xlabel('Days')
    plt.ylabel('Rainall mm')
    plt.legend(loc='best')
    plt.show()
def evaluate_model(trained_model, x_test, y_test, lbl):
    ll_ob = trained_model.predict(x_test, y=y_test)

    return roc_metric(ll_ob, lbl)

def combine_models(trained_models, df, y_target,y_inj, lable, log_vote=True, plot=True):
    #y_test = asmatrix(df[y_target])
    #y_inj, lable = synthetic_fault(asmatrix(df[y_target]))
    y_test = asmatrix(y_inj)
    log_pred = {}
    linear_pred = {}
    #print "Number of models", trained_models.keys()
    for col in trained_models:
        model = trained_models[col]
        x_test = df[col].as_matrix().reshape(-1, 1)
        log_pred[col]  =  model.log_reg.predict_proba(x_test)[:,1].reshape(-1,1)
        linear_pred[col] = model.linear_reg.predict(np.log(x_test + model.eps)).reshape(-1,1)

    a_log_pred = np.mean(np.hstack(log_pred.values()), axis=1)

    a_linear_pred = np.mean(np.hstack(linear_pred.values()), axis=1)
    ll = []
    for _, model in trained_models.iteritems():
        if log_vote:

            result = model.mixl(y_test, a_log_pred, a_linear_pred)
        else:
            result = model.mixl(y_test, log_pred[_], linear_pred[_])
        ll.append(result)
    combined_result = np.mean(np.hstack(ll), axis=1)
    #print combined_result
    if plot:
        test_plot(combined_result, lable, y_test)
    return roc_metric(combined_result, lable)

def test_plot(ll_test, t_lbl, y):

    ix = np.argsort(ll_test, axis=0)[-10:]
    injx = np.where(t_lbl == 1.0)[0]
    plt.plot(y, '.b', label="observations")
    plt.plot(injx, y[injx], '.r', label="injected faults")
    plt.plot(ix, y[ix].reshape(-1), 'oy', mfc='none', label="detected faults")
    #plt.ylabel('Rain (mm)')
    #plt.xlabel("Days")
    plt.legend(loc='best', prop={'size':6})


def test(trained_model, target_station, k_station, test_data, y_inj, t_lbl, plot=False):

    y, x = asmatrix(test_data[target_station]), test_data[k_station].as_matrix()
    #y_inj, t_lbl = synthetic_fault(y, plot=True, f_type="Both", alpha=ALPHA)
    ll_test = trained_model.predict(x=x, y=y_inj)
    if plot:
        test_plot(ll_test, t_lbl, asmatrix(y_inj))

    return roc_metric(ll_test, t_lbl, False)


def train( train_data,target_station="TA00020", num_k=5, pairwise=True, ridge_alpha=0.0):

    k_station = nearby_stations(target_station, k=num_k)
    y, x = train_data[target_station].as_matrix().reshape(-1,1), train_data[k_station].as_matrix()
    # single joint model.


    if pairwise:

        model = MixLinearModel(linear_reg=Ridge(alpha=ridge_alpha))
        model.fit(x=x, y=y)
        # y_inj, lable = synthetic_fault(train_data[target_station])
        # ll_ob = model.predict(x, y=y_inj)
        # print roc_metric(ll_ob, lable)
        return model, k_station
    else:
        # Build pairwise regression model.
        models = {}
        for stn in k_station:

            x_p = train_data[stn].as_matrix().reshape(-1,1)
            model = MixLinearModel(linear_reg=Ridge(alpha=ridge_alpha))
            models[stn] = model.fit(x_p, y)
        #training_prediction = [mdl.predict(asmatrix(train_data[stn]), y=y_inj) for stn, mdl in models.iteritems()]
        return models, k_station

from collections import OrderedDict
train_data = pd.read_csv('tahmostation2016.csv')
test_data = pd.read_csv('tahmostation2017.csv')


def regularization_test(target_station="TA00069"):
    #target_station = "TA0069"
    # result = {}
    # result['station'] = target_station
    # result['num_k'] = K
    # result['anom'] = ALPHA

    y_train, lbl_train = synthetic_fault(train_data[target_station], True, alpha=ALPHA, f_type=FAULT_TYPE)
    y_test, lbl_test = synthetic_fault(test_data[target_station], True, alpha=ALPHA, f_type=FAULT_TYPE)
    alpha_ranges = [0.0, 0.3, 0.9, 30, 1e2, 1e3, 1e4, 1e5]
    results = []
    for alpha in alpha_ranges:
        result= {}
        model, k_station = train(target_station=target_station, num_k=K, train_data=train_data, ridge_alpha=alpha)
        result['alpha'] = alpha
        result['train_auc'] = test(model, target_station=target_station, k_station=k_station,
                                               test_data=train_data, y_inj=y_train, t_lbl=lbl_train, plot=False)
        result['test_auc'] = test(model, target_station=target_station, k_station=k_station,
                                              test_data=test_data,
                                              y_inj=y_test, t_lbl=lbl_test, plot=False)
        results.append(result)
    return results


def main_test(target_station, save_fig=True):
    ## Experiment on the station performance.

    #target_station = "TA00077"
    # joint detection.
    train_result = {}
    train_result['station'] = target_station
    train_result['num_k'] = K
    train_result['anom']  = ALPHA
    #insert synthetic faults
    test_result = train_result.copy()
    plt.subplot(211)
    plt.title(target_station)
    plt.xlabel('2016')
    y_train, lbl_train = synthetic_fault(train_data[target_station], True, alpha=ALPHA, f_type=FAULT_TYPE)
    plt.subplot(212)
    plt.xlabel('2017')
    y_test, lbl_test = synthetic_fault(test_data[target_station], True, alpha=ALPHA, f_type=FAULT_TYPE)
    if save_fig:
        plt.savefig("plots/"+target_station+".jpg")
    plt.close()

    plt.subplot(321)
    plt.title('Training')
    model, k_station = train(target_station=target_station, num_k=K, train_data=train_data)
    print "Training accuracy"
    train_result['train_joint_auc'] = test(model, target_station=target_station, k_station=k_station,
                                                  test_data=train_data, y_inj=y_train, t_lbl=lbl_train, plot=True)
    plt.ylabel('Joint stations',fontsize=6)
    plt.subplot(322)
    plt.title('Test')
    print "Test data set accuracy"
    train_result['test_joint_auc'] = test(model, target_station=target_station, k_station=k_station, test_data=test_data,
                                          y_inj=y_test, t_lbl=lbl_test, plot=True)
    #


    # pairwise:
    print "\nPairwise result"
    models, k_station = train(target_station=target_station, num_k=K, train_data=train_data, pairwise=False)
    print "Training accuracy"
    plt.subplot(323)
    plt.ylabel("Pairwise avg. Logistic vote", fontsize=6)
    train_result['train_pairwise_vote_auc'] = combine_models(models, train_data, target_station, y_train, lbl_train, log_vote=True)
    plt.subplot(325)
    plt.ylabel("Pairwise avg ll", fontsize=6)
    train_result['train_pairwise_auc'] = combine_models(models, train_data, target_station, y_train, lbl_train, log_vote=False)
    print "Testing accuracy "
    plt.subplot(324)
    train_result['test_pairwise_vote_auc'] = combine_models(models, test_data, target_station, y_test, lbl_test, log_vote=True)
    plt.subplot(326)
    train_result['test_pairwise_auc'] = combine_models(models, test_data, target_station, y_test, lbl_test, log_vote=False)
    #plt.show()
    plt.savefig("detectionplot/"+target_station+".jpg")
    plt.close()
    return train_result


def main():

    df = pd.read_csv('sampledata.csv')


    x, y = df.ix[:, 2:].as_matrix(), df.ix[:,1:2].as_matrix()



    # zero-one label
    print "Training sets."
    y_binary = (y>0.0).astype(int)
    model = MixLinearModel(linear_reg=Ridge(alpha=0.5))
    model.fit(x=x, y=y)

    dt, lbl = synthetic_fault_flatline(y, True)
    ll_ob = model.predict(x, y=dt)
    print roc_metric(ll_ob, lbl, False)



    ## Join stations
    models = {}
    colmn = df.columns[2:]
    roc = {}
    predictions = []
    for col in colmn:
        train_col = df[col].as_matrix().reshape(-1, 1)
        models[col] = MixLinearModel(linear_reg=Ridge(alpha=0.5)).fit(x=train_col, y=y)
        # plt.subplot(3,2,2)

        predictions.append(models[col].predict(train_col, y=dt))
        roc[col] = evaluate_model(models[col], train_col, dt, lbl)
    pred = np.hstack(predictions)
    print "AUC of average likelihood"
    print roc_metric(np.sum(pred, axis=1), lbl)
    print "AUC of individual stations"
    print roc

    result = combine_models(models, df, dt)
    print roc_metric(result, lbl)
    print "Experiment on testing data."
    # Testing.
    test_data = pd.read_csv('sampletahmo_test.csv')
    x_t, y_t = test_data.ix[:, 2:].as_matrix(), test_data.ix[:, 1:2].as_matrix()
    y_insert, t_lbl = synthetic_fault(y_t)
    ll_test = model.predict(x=x_t, y=y_insert)
    print roc_metric(ll_test, t_lbl, False)

    print "AUC of test dataset for 2017"
    # #test_roc = roc_metric(ll_test, t_lbl, plot=True)
    roc_test = {}
    t_predictions = []
    for col in colmn:
        roc_test[col] = evaluate_model(models[col], test_data[col].as_matrix().reshape(-1, 1), y_t, t_lbl)
        t_predictions.append(models[col].predict(test_data[col].as_matrix().reshape(-1, 1), y=y_t))
    # ll_aggregate =
    print roc_metric(np.mean(np.hstack(t_predictions), axis=1), t_lbl)
    print "AUC of individual stations "
    print roc_test

    ## Combined model
    result = combine_models(models, test_data, y_t)
    print roc_metric(result, t_lbl)





# if  __name__ == '__main__':
#     random_rainy_days(7)


#
if __name__ == '__main__':
    # Parameters
    FAULT_TYPE = 'BOTH'  # could be 'Spike','Flat', or 'Both'
    K = 5
    ALPHA = 0.02
    ridge_alpha = 0.0
    #main()
    #print nearby_stations('TA00020')
    all_stations = train_data.columns.tolist()
    #print all_stations
    target_station = "TA00025"
    print main_test(target_station, save_fig=True)
    #

    # all_auc = []
    # for target_station in all_stations[:]:
    #     all_auc += [main_test(target_station, save_fig=False)]
    # results = pd.DataFrame(all_auc)
    # results.to_csv("k_"+str(K)+"_results.csv",index=False)




    # Regularization expriments
    #target_station = "TA00025"
    #reg = pd.DataFrame(regularization_test(target_station))
    #reg.to_csv(target_station+"regularization.csv", index=False)

    #model, k_station = train(target_station=target_station, num_k=5)
    #test(model, target_station=target_station, k_station=k_station)

