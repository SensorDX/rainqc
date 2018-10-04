import matplotlib.pylab as plt
import sklearn.metrics as mt
from numpy import round
def roc_metric(pred, obs, plot=False):
    fpr_rt_lm, tpr_rt_lm, _ = mt.roc_curve(obs, pred)
    auc_score = mt.auc(fpr_rt_lm, tpr_rt_lm, reorder=True)
    if plot:

        plt.clf()
        plt.plot(fpr_rt_lm, tpr_rt_lm, label='ROC curve')
        plt.ylabel("tpr")
        plt.xlabel("fpr")
        plt.plot(range(0, 2), range(0, 2), '-r')
        plt.text(0.6, 0.2, "auc=" + str(auc_score))
        plt.legend(loc='best')
        plt.show()
    return round(auc_score,3)

def precision_recall(pred, obs, plot=False):
    # Precision at @k recall,
    mt.precision_recall_curve(obs, pred)

#
# def insert_faults():
#     abnormal_report = range(200, 210)  # large abnormal rainfall report.
#     rainy_days = range(107, 117)
#     insert_fault[abnormal_report] = 20.0
#     insert_fault[rainy_days] = 0.0  # Flatten days with rain events.
#     fault_days = abnormal_report + rainy_days