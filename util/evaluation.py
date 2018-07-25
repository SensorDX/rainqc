import matplotlib.pylab as plt
import sklearn.metrics as mt

def roc_metric(pred, obs):
    fpr_rt_lm, tpr_rt_lm, _ = mt.roc_curve(obs, pred)
    auc_score = mt.auc(fpr_rt_lm, tpr_rt_lm, reorder=True)

    plt.clf()
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='ROC curve')
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.plot(range(0, 2), range(0, 2), '-r')
    plt.text(0.6, 0.2, "auc=" + str(auc_score))
    plt.legend(loc='best')
    return auc_score
