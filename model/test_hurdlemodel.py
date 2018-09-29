
from hurdle_regression import MixLinearModel
import pandas as pd
from util.evaluation import roc_metric
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
import matplotlib.pylab as plt
import numpy as np


# input data

df = pd.read_csv('sampledata.csv')
x, y = df.ix[:, 2:].as_matrix(), df.ix[:,1:2].as_matrix()
print x.shape ,y.shape
# zero-one label
y_binary = (y>0.0).astype(int)
model = MixLinearModel()

# fit logistic regression model to 0/1 model
# logit = LogisticRegression(fit_intercept=False, C=1e9)
# logit.fit(y=y_binary, X=x)
# print logit.coef_
# #evaluate log model
# print ("Logistic regression model")
# lr_fitted = logit.predict_proba(x)[:,1]
# print lr_fitted[1:10]
# #roc_metric(lr_fitted, y_binary)  :passed
# print("## Using statsmodels ")
# glm = sm.GLM(y_binary,x,family=sm.families.Binomial(sm.families.links.logit)).fit()
# print glm.summary()
# glm_fitted = glm.fittedvalues
# print glm_fitted[1:10]
# glm.save(open("test.glm",'w'))

"""
The key difference b/n the glm of the statsmodels and logisticregression of sklearn is, by defualt the sklearn
prediction outputs binary value, while the glm outputs probability.
"""
##
# observed_value = x.mean(axis=1)
# lreg = LinearRegression()
# lreg.fit(X=np.log(x+ model.eps), y=np.log(x+model.eps), sample_weight=glm_fitted)
# lreg_fitted = lreg.predict(np.log(x+model.eps))
# res = (lreg_fitted - np.log(model.eps + y))
#model.residual_plot(np.log(observed_value + model.eps), np.log(y+model.eps), lreg_fitted)

## Using the hurdel predict the fault level of a given stations, given the other stations.

model.fit(x=x, y=y)
fitted_value = model.predict(x,y)

#roc_metric( model.log_reg.predict_proba(x)[:,1], y_binary) # debugging
# insert faults to the
def synthetic_fault(observations, plot=False):
    dt = observations.copy()
    abnormal_report = range(200, 210)
    rainy_days = range(107, 117)
    dt[abnormal_report] = 20.0
    dt[rainy_days] = 0.0
    faulty_day = abnormal_report + rainy_days
    lbl = np.zeros([dt.shape[0]])
    lbl[faulty_day] = 1.0
    return dt, lbl

plt.subplot(321)
dt, lbl = synthetic_fault(y, True)
ll_ob = model.predict(x,y=dt)
print roc_metric(ll_ob, lbl, False)
#model.residual_plot(np.log(observed_value+model.eps), np.log(y+model.eps), fitted_value)
#print roc_metric()
#yhat = -np.log(model.predict(x, y))

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
## Train using pairwise.

## Using test data from another years.
test_data  = pd.read_csv('sampletahmo_test.csv')
x_t, y_t = test_data.ix[:, 2:].as_matrix(), test_data.ix[:,1:2].as_matrix()
#insert faults on the test
y_insert, t_lbl = synthetic_fault(y_t)
print x_t.shape
ll_test = model.predict(x=x_t, y=y_insert)
print roc_metric(ll_test, t_lbl, False)

models ={}
colmn = df.columns[2:]
roc = {}
predictions = []
for col in colmn:
    train_col = df[col].as_matrix().reshape(-1, 1)
    models[col] = MixLinearModel().fit(x=train_col, y=y)
    #plt.subplot(3,2,2)
    predictions.append(models[col].predict(train_col, y=dt))
    roc[col] = evaluate_model(models[col], train_col, dt, lbl)
pred = np.hstack(predictions)
print "AUC of average likelihood"
print roc_metric(np.sum(pred, axis=1), lbl)
print "AUC of individual stations"
print roc

print "AUC of test dataset for 2017"
# #test_roc = roc_metric(ll_test, t_lbl, plot=True)
roc_test = {}
t_predictions = []
for col in colmn:
     roc_test[col] = evaluate_model(models[col], test_data[col].as_matrix().reshape(-1,1), y_t, t_lbl)
     t_predictions.append(models[col].predict(test_data[col].as_matrix().reshape(-1,1), y=y_t))
#ll_aggregate =
print roc_metric(np.mean(np.hstack(t_predictions), axis=1), t_lbl)
print "AUC of individual stations "
print roc_test