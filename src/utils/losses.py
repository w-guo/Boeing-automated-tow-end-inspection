import keras.backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred, bce=0.5, dice=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_loss(
        y_true, y_pred) * dice


def bce_dice_loss_non_empty(y_true, y_pred):
    """
    Return bce_dice_loss when max pixel = 1 (i.e.the image is non-empty) and 0 when the image is empty
    y_true format: (batch, h, w, channel)
    """
    return K.max(K.max(y_true, axis=1), axis=1) * bce_dice_loss(
        y_true, y_pred, bce=0.5, dice=0.5)


def get_iou_vector(A, B):
    # This is the iou metric used by Kaggle competition: https://www.kaggle.com/c/tgs-salt-identification-challenge
    # It "grades" the real iou. With this metric, if real iou < 0.5 (usually, iou >= 0.5 is
    # considered good), it got grade 0.
    # Likewise, if 0.5 <= real iou < 0.55, iou grade = 0.1; ... if real iou > 0.95, iou grade = 1
    # https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union  # real iou

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def iou(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def iou_lovasz(label, pred):
    # lovasz_loss need input range (-∞，+∞), so the default threshod
    # for pixel prediction is 0 instead of 0.5
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)


def lovasz_loss(y_true, y_pred):
    """
    Lovasz-Softmax and Jaccard hinge loss in Tensorflow
    https://github.com/bermanmaxim/LovaszSoftmax
    """
    y_true, y_pred = K.cast(K.squeeze(y_true, -1),
                            'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    # logits = K.log(y_pred / (1. - y_pred)) # the original code
    # https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
    logits = y_pred
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss


def symmetric_lovasz_loss(y_true, y_pred):
    # https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053#406866
    y_true, y_pred = K.cast(K.squeeze(y_true, -1),
                            'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    logits = y_pred
    loss = (lovasz_hinge(logits, y_true) +
            lovasz_hinge(-logits, 1 - y_true)) / 2
    return loss


# --------------------------- Binary Lovasz hinge loss ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:

        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)

        # Fixed python3
        losses.set_shape((None, ))

        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors,
                                          k=tf.shape(errors)[0],
                                          name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        # loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        # ELU + 1
        loss = tf.tensordot(tf.nn.elu(errors_sorted) + 1.,
                            tf.stop_gradient(grad),
                            1,
                            name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss")
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1, ))
    labels = tf.reshape(labels, (-1, ))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels
