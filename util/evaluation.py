import matplotlib.pylab as plt
import sklearn.metrics as mt
from numpy import round
def roc_metric(pred, obs, plot=False, plt=None):
    fpr_rt_lm, tpr_rt_lm, _ = mt.roc_curve(obs, pred)
    auc_score = mt.auc(fpr_rt_lm, tpr_rt_lm, reorder=True)
    if plot:
        #plt.clear()


        plt.plot(fpr_rt_lm, tpr_rt_lm, label='ROC curve')
        plt.ylabel("tpr")
        plt.xlabel("fpr")
        plt.plot(range(0, 2), range(0, 2), '-r')
        plt.text(0.6, 0.2, "auc=" + str(auc_score))
        plt.legend(loc='best')
        return plt
    return round(auc_score,3)
def plot_pr_recall(pr, recall, plt):
    plt.plot(recall, pr, label='pr curve')
    plt.ylabel("pr")
    plt.xlabel("recall")
    plt.plot(range(0, 2), range(0, 2), '-r')
    #plt.text(0.6, 0.2, "auc=" + str(auc_score))
    plt.legend(loc='best')


def precision_recall(pred, obs, plot=False):
    # Precision at @k recall,
    pr, recall, _ = mt.precision_recall_curve(obs, pred)
    if plot:
        return plot_pr_recall(pr, recall)
    return pr, recall
def ap(pred, obs):
    return mt.average_precision_score(obs, pred)

def precision_at_recall(pred, obs):
    pr, recall, _ = mt.precision_recall_curve(obs, pred)
    return pr, recall

#
# def insert_faults():
#     abnormal_report = range(200, 210)  # large abnormal rainfall report.
#     rainy_days = range(107, 117)
#     insert_fault[abnormal_report] = 20.0
#     insert_fault[rainy_days] = 0.0  # Flatten days with rain events.
#     fault_days = abnormal_report + rainy_days
if __name__ == '__main__':
    precision_at_recall([1, 0, 1, 1], [.32, .52, .26, .86])
