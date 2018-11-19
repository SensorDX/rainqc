import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import pygal


FAULT_FLAG = 1.0
FLAT_LINE = 0.0


def build_graph(x_coordinates, y_coordinates):
    img = io.BytesIO()
    plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')

    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def build_metric_graph(pred, lbl, plt_type='ap'):
    #plt = plot_curve(pred, lbl, plt_type)
    if plt_type=='ap':
        pr, rc,_ = precision_recall_curve(lbl, pred)
        ap = np.round(average_precision_score(lbl, pred),2)
        plt.plot(rc, pr,label='AP: '+str(ap))
        plt.ylabel('PR')
        plt.xlabel('Recall')
    elif plt_type =='auc':
        fpr, tpr,_ = roc_curve(lbl, pred)
        roc = np.round(auc(fpr, tpr, reorder=True),2)
        plt.plot(fpr, tpr, label='AUC: '+str(roc))
        plt.plot(range(0, 2), range(0, 2), '-r')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
    else:
        return
    plt.legend(loc='best')
    img = io.BytesIO()
    #plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

# Plot operations

def out_plot(scores, date_range, target_station, flag_data=None):
    line_map = [(dt.date(), sc) for dt, sc in zip(date_range, scores)]

    graph = pygal.DateLine(x_label_rotation=35, stroke=False, human_readable=True)  # disable_xml_declaration=True)
    graph.force_uri_protocol = 'http'
    graph.title = '{}: Score.'.format(target_station)
    graph.add(target_station, line_map)
    if flag_data is not None:
        flag_label = flag_data==FAULT_FLAG
        fault_line = [ line_map[ix] if flag_label[ix] else (None,None) for ix in range(len(flag_label))]

        graph.add('Faults', fault_line)

    graph_data = graph.render_data_uri()  # is_unicode=True)
    return graph_data