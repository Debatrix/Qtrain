import os
import numpy as np
import torch
import sklearn.metrics
from scipy.special import softmax

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def curve2text(curve):
    text = ''
    for k, v in curve.items():
        text += '{}:'.format(k.upper())
        for x in v:
            text += '{:.2f}, '.format(x)
        text += ';'
    return text


# ##############################################################################
def plot_log_roc(writer, roc, epoch=0, log_name='roc'):
    fpr, tpr, thr, auc = roc
    figure = plt.figure()
    plt.plot(fpr, tpr, 'r-')
    plt.xlabel('False positive')
    plt.ylabel('True positive')
    plt.legend(labels=['AUC:{:.4f}'.format(auc)])
    plt.grid()
    l = np.linspace(0, 1, fpr.shape[0])
    plt.plot(l, l, ':', color='gray')
    writer.add_figure(log_name, figure, epoch)
    curve_text = curve2text({'fpr': fpr, 'tpr': tpr, 'thr': thr, 'auc': [auc]})
    writer.add_text(log_name, curve_text)


def plot_log_prc(writer, prc, epoch=0, log_name='pr_curve'):
    precision, recall, thr = prc
    figure = plt.figure()
    plt.plot(recall, precision, 'r-')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.grid()
    writer.add_figure(log_name, figure, epoch)
    curve_text = curve2text({
        'precision': precision,
        'recall': recall,
        'thr': thr
    })
    writer.add_text(log_name, curve_text)


def plot_feature(writer, feature, label, method='PCA', epoch=0):
    figure = plt.figure()
    if method.lower() == 'tsne':
        projector = TSNE(n_components=2, init='pca', perplexity=60)
        method = 'tSNE'
    else:
        projector = PCA(n_components=2)
        method = 'PCA'
    d_feature = projector.fit_transform(feature)
    label_set = list(set(label.tolist()))
    colors = ["red", "blue", "brown", "orange", "pink", "yellow", "green"]
    for i, l in enumerate(label_set):
        plt.scatter(d_feature[np.where(label == l), 0],
                    d_feature[np.where(label == l), 1],
                    color=colors[i],
                    label=l,
                    edgecolor='none',
                    alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    writer.add_figure("feature/{}".format(method), figure, epoch)


# ##############################################################################


def bin_predict_metrics(label, proba, prediction, level='tile'):
    auc = sklearn.metrics.roc_auc_score(label, proba)
    roc = sklearn.metrics.roc_curve(label, proba)
    pr = sklearn.metrics.precision_recall_curve(label, proba)
    recall = sklearn.metrics.recall_score(label, prediction)
    precision = sklearn.metrics.precision_score(label, prediction)
    f1 = sklearn.metrics.f1_score(label, prediction)
    acc = sklearn.metrics.accuracy_score(label, prediction)
    result = {
        '{}_recall'.format(level): recall,
        "{}_precision".format(level): precision,
        "{}_f1".format(level): f1,
        "{}_auc".format(level): auc,
        "{}_acc".format(level): acc,
        "{}_pr".format(level): pr,
        "{}_roc".format(level): [*roc, auc],
    }
    return result


def tile_predict_metrics(label, proba, prediction):
    if len(label.shape) != 2 and np.max(label) <= 1:
        result = bin_predict_metrics(label, proba, prediction, 'tile')
    else:
        raise NotImplemented
    return result


def slide_predict_metrics(tile_label,
                          tile_proba,
                          tile_prediction,
                          instance,
                          type='count'):
    inset = list(set(instance))
    instance = np.array(instance)

    if len(tile_label.shape) != 2 and np.max(tile_label) <= 1:
        proba = np.zeros((len(inset), 2))
        label = np.zeros((len(inset), ))

        for idx, ins in enumerate(inset):
            if type == 'count':
                p = (tile_prediction[np.where(instance == ins)]
                     == 1).sum() / len(np.where(instance == ins)[0])
                proba[idx] = np.array((1 - p, p))
            else:
                p = tile_proba[np.where(instance == ins)]
                proba[idx] = p.mean(0)
            label[idx] = tile_label[np.where(instance == ins)][0]

        proba = softmax(proba, 1)
        prediction = np.argmax(proba, 1)
        proba = proba[:, 1]
        result = bin_predict_metrics(label, proba, prediction, 'slide')
    else:
        raise NotImplemented
    return result


# ##############################################################################


def evaluation(val_save, val_num):
    result = {}
    label = np.concatenate(val_save['label'], axis=0)
    instance = val_save['instance']
    if 'pred_loss' in val_save:
        result['pred_loss'] = val_save['pred_loss'] / val_num
    if 'proba' in val_save:
        proba = np.concatenate(val_save['proba'], axis=0)
        prediction = np.concatenate(val_save['prediction'], axis=0)
    else:
        proba = np.concatenate(val_save['prediction'], axis=0)
        proba = softmax(proba, 1)
        prediction = np.argmax(proba, 1)
        proba = proba[:, 1]

    if 'feature' in val_save:
        feature = np.concatenate(val_save['feature'], 0)
        result['ch_index'] = sklearn.metrics.calinski_harabasz_score(
            feature, label)

    result.update(tile_predict_metrics(label, proba, prediction))
    val_save['tile_roc'] = result.pop('tile_roc')
    val_save['tile_pr'] = result.pop('tile_pr')

    result.update(slide_predict_metrics(label, proba, prediction, instance))
    val_save['slide_roc'] = result.pop('slide_roc')
    val_save['slide_pr'] = result.pop('slide_pr')

    return result


def val_plot(log_writer, epoch, val_save, mode='val'):
    label = np.concatenate(val_save['label'], axis=0)
    feature = np.concatenate(val_save['feature'], 0)

    plot_log_roc(
        log_writer,
        val_save['tile_roc'],
        epoch,
        log_name="tile_roc",
    )
    plot_log_roc(
        log_writer,
        val_save['slide_roc'],
        epoch,
        log_name="slide_roc",
    )
    plot_log_prc(
        log_writer,
        val_save['tile_pr'],
        epoch,
        log_name='tile_pr',
    )
    plot_log_prc(
        log_writer,
        val_save['slide_pr'],
        epoch,
        log_name='slide_pr',
    )
    plot_feature(
        log_writer,
        feature,
        label,
        method='PCA',
        epoch=epoch,
    )

    log_writer.add_embedding(feature, val_save['tag'], global_step=epoch)