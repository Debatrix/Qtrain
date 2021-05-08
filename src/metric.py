import torch
import numpy as np
from sklearn.metrics import roc_curve, auc


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
def _reshape_dfs(dfs):
    dfs = dfs.reshape(-1)
    return dfs


def _reshape_name(name):
    name = name[1:, :].T.reshape(-1)
    return name


def _name2index(name, name_dict, label_dict):
    index = [name_dict[x] for x in name if x in name_dict]  # TODO
    label = [label_dict[x] for x in index]
    return np.array(index), np.array(label)


def get_sim_mat(sim, name):
    index, label = _name2index(name, sim['name'], sim['label'])
    sim_mat = sim['sim'][index, :][:, index]
    return sim_mat, label


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


def _onehot(labels):
    # get one hot vectors, from x as a list of strings
    classes = tuple(set(labels))
    n_class = len(classes)
    indx = [classes.index(sample) for sample in labels]
    onehot = np.eye(n_class, dtype=np.int).take(np.array(indx), 0)
    return onehot  # N X D


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

    FAR, TAR, T = roc_curve(signals, scores)
    FRR = 1 - TAR
    return TAR, FAR, FRR, T


def IQAMetric(sim, score, name, bins=10):
    score = _reshape_dfs(score)
    name = _reshape_name(name)

    EERs, IRRs = [], []
    for qths in np.arange(score.min(), score.max(),
                          (score.max() - score.min()) / bins):
        _name = name[score > qths]
        sim_mat, label = get_sim_mat(sim, _name)
        try:
            TAR, FAR, FRR, T = get_roc_curve(*get_scores(sim_mat, label))
            EER, T_eer = get_eer(FAR, FRR, T)
        except:
            EER = 0
        EERs.append(EER)
        IRRs.append(1 - len(label) / len(name))
    EERs = np.array(EERs)
    IRRs = np.array(IRRs)
    auc = ((np.append(IRRs, 1)[1:] - IRRs) * EERs).sum()

    return IRRs, EERs, auc


# ##############################################################################
if __name__ == "__main__":
    metric = SegmentationMetric(2)
    gt = torch.randint(2, (2, 3072 // 8, 4080 // 8))
    prob = torch.softmax(torch.rand((2, 3, 3072 // 8, 4080 // 8)), 1)
    metric.update(prob, gt)
    print(metric.get())
