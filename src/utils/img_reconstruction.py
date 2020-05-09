import cv2
import gc
import numpy as np
from pre_processing import pre_process

# https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/smooth_tiled_predictions.py
# Modifications are made on the function "_windowed_subdivs" from the original code


def rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), 1))
    mirrs.append(np.rot90(np.array(im), 2))
    mirrs.append(np.rot90(np.array(im), 3))
    im = np.fliplr(im)
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), 1))
    mirrs.append(np.rot90(np.array(im), 2))
    mirrs.append(np.rot90(np.array(im), 3))

    return mirrs


def rotate_mirror_undo(im_mirrs):
    """
    Merge a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), 3))
    origs.append(np.rot90(np.array(im_mirrs[2]), 2))
    origs.append(np.rot90(np.array(im_mirrs[3]), 1))

    origs.append(np.fliplr(np.array(im_mirrs[4])))
    origs.append(np.fliplr(np.rot90(np.array(im_mirrs[5]), 3)))
    origs.append(np.fliplr(np.rot90(np.array(im_mirrs[6]), 2)))
    origs.append(np.fliplr(np.rot90(np.array(im_mirrs[7]), 1)))

    return np.mean(origs, axis=0)


def windowed_subdivs(model, input_ch, train_mean, train_std, padded_img,
                     window_size, overlap_pct):
    """
    Create tiled overlapping patches.

    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )

    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """

    step = int(window_size * (1 - overlap_pct))
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    if input_ch == 3:
        for i in range(0, padx_len - window_size + 1, step):
            subdivs.append([])
            for j in range(0, pady_len - window_size + 1, step):
                patch = padded_img[i:i + window_size, j:j + window_size]
                patch = pre_process(patch, train_mean, train_std) / 255
                patch = cv2.merge((patch, patch, patch))
                subdivs[-1].append(patch)
    else:
        for i in range(0, padx_len - window_size + 1, step):
            subdivs.append([])
            for j in range(0, pady_len - window_size + 1, step):
                patch = padded_img[i:i + window_size, j:j + window_size]
                patch = pre_process(patch, train_mean, train_std) / 255
                patch = np.expand_dims(patch, axis=-1)
                subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    subdivs = model.predict(subdivs)
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, 1)
    gc.collect()

    return subdivs


def recreate_from_subdivs(subdivs, window_size, overlap_pct, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size * (1 - overlap_pct))
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len - window_size + 1, step):
        b = 0
        for j in range(0, pady_len - window_size + 1, step):
            windowed_patch = subdivs[a, b]
            y[i:i + window_size, j:j +
              window_size] = y[i:i + window_size, j:j +
                               window_size] + windowed_patch[:, :, 0]
            b += 1
        a += 1

    for i in range(step, padx_len - window_size + 1, step):
        y[i:int(i + window_size * overlap_pct
                ), :] = y[i:int(i + window_size * overlap_pct), :] / 2

    for j in range(step, pady_len - window_size + 1, step):
        y[:, j:int(j + window_size * overlap_pct
                   )] = y[:, j:int(j + window_size * overlap_pct)] / 2

    return y