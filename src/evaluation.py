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


# ##############################################################################
# author: @liminn
# page: https://github.com/liminn/ICNet-pytorch/blob/master/utils/metric.py
class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (1e-16 + self.total_label
                                             )  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (1e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    pixel_correct = 0
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    try:
        pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    except:
        print("predict size: {}, target size: {}, ".format(
            predict.size(), target.size()))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1  # [N,H,W]
    target = target.float() + 1  # [N,H,W]

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(),
                             bins=nbins,
                             min=mini,
                             max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item(
    ) == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection,
                                          bins=numClass,
                                          range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k],
                       minlength=num_cls**2).reshape(num_cls,
                                                     num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


# ##############################################################################
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


def plot_log_dis(writer, dfs, epoch=0, log_name='distribution'):
    dfs = np.array([x for x in dfs.values()])
    figure = plt.figure()
    n, bins, patches = plt.hist(dfs,
                                30,
                                density=True,
                                facecolor='r',
                                alpha=0.25)
    d = np.sort(dfs)
    y = ((1 / (np.power(2 * np.pi, 0.5) * d.std())) * np.exp(-0.5 * np.power(
        (d - d.mean()) / d.std(), 2)))
    plt.plot(d, y, 'r-')
    yy = 0.5 * (1 + erf(((d - dfs.mean()) / (dfs.std() * np.sqrt(2)))))
    plt.plot(d, yy, 'g-')
    plt.grid(True)
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

    return result


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

    return result


def q_val_plot(log_writer, epoch, val_save, mode='val'):
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


def r_val_plot(log_writer, epoch, val_save, mode='val'):
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
