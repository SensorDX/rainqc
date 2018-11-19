from collections import defaultdict
import numpy as np
from src.common.utils import sample
from src.common.evaluation import ap, roc_metric
import src.common.evaluation as metric
FAULT_FLAG = 1.0
FLAT_LINE = 0.0

def group_events(series, threshold = 1.0, filter_length=0):
    group = defaultdict(int)
    group_num = []
    g = 0
    for i, ix in enumerate(series):
        if ix > threshold:
            group_num.append(i)
        else:
            if len(group_num) > 0:
                group[g] = group_num
                group_num = []
                g += 1
    if filter_length>0:
        group = { key:value for key, value in group.iteritems() if len(value)>filter_length}
    return group

def synthetic_groups(observations, plot=False, alpha=0.1, threshold=1.0, fault_type=FLAT_LINE):
    """

    Args:
        observations (nd.array):
        plot:
        alpha:
        threshold:
        fault_type:

    Returns:

    """

    num_faults = int(np.ceil(alpha * len(observations)))
    dt = observations.copy()
    lbl = np.zeros([dt.shape[0]])
    rainy_days = np.where(dt > threshold)[0]
    injected_indx = []
    done = False
    injected_group = {}

    if len(rainy_days) < num_faults:
        raise NameError("No enough rainy days for station")
    rainy_events = group_events(dt, threshold, filter_length=1)
    if len(rainy_events)<1:
        raise NameError("No enough rainy days for the station")
    selected_group = sample(rainy_events.keys())

    for g in selected_group:
        i = 0
        for i, ix in enumerate(rainy_events[g]):

            if len(injected_indx) >=num_faults:
                done = True
                break
            injected_indx.append(ix)
        if i>0:
            injected_group[g] = FAULT_FLAG
        if done:
            break

    dt[injected_indx] = FLAT_LINE
    lbl[injected_indx] = FAULT_FLAG

    # if plot:
    #     plot_synthetic(observations, dt, injected_indx)
    groups = {"data":dt, "label":lbl, "injected_group": injected_group, "group_events":rainy_events}

    return groups #dt, groups, lbl
def evaluate_groups(groups, ll_score):


    injected_group = groups["injected_group"].keys()
    mx_ll_score = ll_score.copy()
    lbl = groups['label']
    for ig in injected_group:
        ix_g = groups["group_events"][ig]
        ix_g = [ix for ix in ix_g if lbl[ix] == 1]
        max_score = np.max(mx_ll_score[ix_g])
        mx_ll_score[ix_g] = max_score
    groups['gp_score'] = mx_ll_score
    train_result ={"point":{},"group":{}}
    #print "with out group"
    train_result["point"]["auc"] = roc_metric(ll_score, lbl)
    train_result["point"]["pr"]  = ap(ll_score, lbl)
    #print "With group"
    train_result["group"]["auc"] = roc_metric(mx_ll_score, lbl)
    train_result["group"]["pr"] = ap(mx_ll_score, lbl)
    return train_result, groups

def plot_curve(pred, lbl, plt='pr'):
    if plt=='pr':
        return metric.precision_recall(pred, lbl, plot=True)

    elif plt=='auc':
        return metric.roc_metric(pred, lbl, plot=True)
    else:
        return

