import itertools

import numpy as np
import sklearn.metrics
from scipy import stats
from scipy.special import softmax, erf

import torch
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


# #############################################################################
def _onehot(labels):
    # get one hot vectors, from x as a list of strings
    classes = tuple(set(labels))
    n_class = len(classes)
    indx = [classes.index(sample) for sample in labels]
    onehot = np.eye(n_class, dtype=np.int).take(np.array(indx), 0)
    return onehot  # N X D


def get_eer(FAR, FRR, T=None):
    # get EER from roc curve
    FAR = np.array(FAR)
    FRR = np.array(FRR)
    gap = np.abs(FAR - FRR)
    index = np.where(gap == np.min(gap))
    EER = FRR[index][0]
    if T is None:
        T_eer = None
    else:
        T = np.array(T)
        T_eer = T[index][0]
    return EER, T_eer


def get_scores(sim_mat, labels):
    # get similarity scores and the signal of pairs for feature
    #
    # sim_mat:  N X N feature matrix, N is the number of samples
    # labels:   N labels of the features
    #
    # scores:   similarity scores, sorted from large to small
    # signals:  signals of pairs, 1 for positive, 0 for negative
    N = sim_mat.shape[0]
    ind_keep = np.logical_not(np.eye(N, dtype=np.bool)).reshape(-1)
    # ind_keep = np.array(np.triu(np.matrix(np.ones((N, N))), 1),
    #                     dtype=np.bool).reshape(-1)

    signals = _onehot(labels)
    signals = np.dot(signals, signals.T).reshape(-1)[ind_keep]
    scores = sim_mat.reshape(-1)[ind_keep]

    return scores, signals


def get_roc_curve(scores, signals):
    # get axis of roc curves, and threshold values
    #
    # scores:     (n_samples*[n_samples-1],) similarity scores, sorted from large to small
    # signals:    (n_samples*[n_samples-1],) signals of pairs, 1 for positive, 0 for negative
    #
    # TAR:        (>2,) true accept rate
    # FAR:        (>2,) false accept rate
    # FRR:        (>2,) false reject rate
    # T:          (n_thresholds,) threshold values

    FAR, TAR, T = sklearn.metrics.roc_curve(signals, scores)
    AUC = sklearn.metrics.auc(FAR, TAR)
    FRR = 1 - TAR
    return TAR, FAR, FRR, AUC, T


def get_dfs(sim, feature, label, name, top=0.25):
    label_set = list(set(label.tolist()))
    name = np.array(name)
    dfs = {}
    for l in label_set:
        intra_index = np.where(label == l)[0]
        intra_name = name[intra_index]
        intra_sim = sim[intra_index, :][:, intra_index]
        intra_feature = feature[intra_index, :]
        intra_score = intra_sim.mean(1)
        g_num = int(top * intra_score.size)
        g_idx = np.argpartition(intra_score, -g_num)[-g_num:]
        center = intra_feature[g_idx].mean(0).reshape(-1, 1)
        intra_dfs = np.dot(intra_feature, center).reshape(-1)
        dfs.update({x: y for x, y in zip(intra_name, intra_dfs)})
    return dfs


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


def plot_log_prc(writer, prc, epoch=0, log_name='pr_curve'):
    precision, recall, thr = prc
    figure = plt.figure()
    plt.plot(recall, precision, 'r-')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.grid()
    writer.add_figure(log_name, figure, epoch)


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


def _plot_dis(figure, dfs, c='r'):
    if isinstance(dfs, dict):
        dfs = np.array([x for x in dfs.values()])

    n, bins, patches = plt.hist(dfs, 30, density=True, facecolor=c, alpha=0.25)
    d = np.sort(dfs)
    y = ((1 / (np.power(2 * np.pi, 0.5) * d.std())) * np.exp(-0.5 * np.power(
        (d - d.mean()) / d.std(), 2)))
    plt.plot(d, y, c + '-')
    # yy = 0.5 * (1 + erf(((d - dfs.mean()) / (dfs.std() * np.sqrt(2)))))
    # plt.plot(d, yy, 'g-')
    plt.grid(True)
    return figure


def plot_log_dis(writer, dfs, epoch=0, log_name='distribution'):
    color = ['r', 'g', 'b', 'y', 'k']
    figure = plt.figure()
    if not isinstance(dfs, list):
        dfs = [dfs]
    for idx in range(len(dfs)):
        figure = _plot_dis(figure, dfs[idx], color[idx])
    writer.add_figure(log_name, figure, epoch)


# ##############################################################################
def recognition_metrics(feature, label):
    result = {}
    sim_mat = np.dot(feature, feature.T)
    TAR, FAR, FRR, AUC, T = get_roc_curve(*get_scores(sim_mat, label))
    EER, T_eer = get_eer(FAR, FRR, T)
    result['sim'] = sim_mat
    result['eer'] = EER
    result['roc'] = (FAR, TAR, T, AUC)
    return result


# ##############################################################################


def r_evaluation(val_save, val_num):
    result = {}
    label = np.concatenate(val_save['label'], axis=0)
    feature = np.concatenate(val_save['feature'], 0)

    if 'pred_loss' in val_save:
        result['pred_loss'] = val_save['pred_loss'] / val_num

    result['ch_index'] = sklearn.metrics.calinski_harabasz_score(
        feature, label)
    result.update(recognition_metrics(feature, label))

    val_save['roc'] = result.pop('roc')
    val_save['sim'] = result.pop('sim')
    val_save['name'] = list(itertools.chain(*val_save['name']))
    val_save['feature'] = feature
    val_save['label'] = label

    val_save['dfs'] = get_dfs(val_save['sim'], val_save['feature'],
                              val_save['label'], val_save['name'])

    return result, val_save


def q_evaluation(val_save, val_num):
    result = {}
    dfs = np.concatenate(val_save['dfs'], axis=0).reshape(-1)
    pdfs = np.concatenate(val_save['prediction'], axis=0).reshape(-1)
    name = np.concatenate(val_save['name'], axis=0).reshape(-1)
    mask = np.concatenate(val_save['mask'], axis=0)
    heatmap = np.concatenate(val_save['heatmap'], axis=0)
    image = np.concatenate(val_save['image'], axis=0)

    if 'pred_loss' in val_save:
        result['pred_loss'] = val_save['pred_loss'] / val_num

    result['srocc'] = stats.spearmanr(pdfs.reshape(-1), dfs.reshape(-1))[0]
    result['lcc'] = stats.pearsonr(pdfs.reshape(-1), dfs.reshape(-1))[0]

    val_save['prediction'] = pdfs
    val_save['dfs'] = dfs
    val_save['name'] = name
    val_save['mask'] = mask
    val_save['heatmap'] = heatmap
    val_save['image'] = image

    val_save['pdfs'] = {name[x]: pdfs[x] for x in range(len(name))}

    return result, val_save


def q_val_plot(log_writer, epoch, val_save):
    idx = np.random.randint(val_save['mask'].shape[0])
    mask = val_save['mask'][idx]
    heatmap = val_save['heatmap'][idx]
    image = val_save['image'][idx][0]

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    mask[mask != 0] = 1
    mask = mask.astype(np.float64)
    heatmap = softmax(heatmap, 0)
    heatmap = heatmap[1:, :, :].sum(0)
    show_mask = np.stack((np.zeros_like(mask), mask, heatmap), axis=0)
    mask += image
    heatmap += image
    image = np.stack((heatmap, mask, image), 0)

    image = np.clip(image, 0, 1)
    show_mask = np.clip(show_mask, 0, 1)
    log_writer.add_image('mask', show_mask, epoch)
    log_writer.add_image('image', image, epoch)

    plot_log_dis(log_writer, [val_save['dfs'], val_save['prediction']], epoch)


def r_val_plot(log_writer, epoch, val_save):
    plot_log_roc(
        log_writer,
        val_save['roc'],
        epoch,
        log_name="roc",
    )
    plot_log_dis(
        log_writer,
        val_save['dfs'],
        epoch,
        # log_name="dfs",
    )

    # tag = [
    #     '{}_{}'.format(*x) for x in zip(val_save['label'], val_save['name'])
    # ]
    log_writer.add_embedding(val_save['feature'],
                             val_save['label'],
                             global_step=epoch)
