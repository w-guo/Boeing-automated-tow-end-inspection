import os
import cv2
import fnmatch
import numpy as np
import params as prm
from utils.extract_lines import lineMagnitude, find_major_orientations, merge_lines, remove_isolated_lines

# %% Compute a list of boundary ranges (boundary position +/- tolerance) of sub-images

tol = 7  # tolerance
cancat_range_x = []
cancat_range_y = []
for i in range(prm.img_w, prm.pad_w, prm.img_w):
    # compute boundary position in x-axis
    subimg_bd_x = int(i - prm.aug_w)
    # compute boundary range in x-axis
    range_x = range(subimg_bd_x - tol, subimg_bd_x + tol)
    cancat_range_x += list(range_x)

for j in range(prm.img_h, prm.pad_h, prm.img_h):
    # compute boundary position in y-axis
    subimg_bd_y = int(j - prm.aug_h)
    # compute boundary range in y-axis
    range_y = range(subimg_bd_y - tol, subimg_bd_y + tol)
    cancat_range_y += list(range_y)

# %% Find class boundary

test_dir = '../data/test/'
test_file_list = fnmatch.filter(os.listdir(test_dir + 'src/'), '*.jpg')

for file in test_file_list:
    src = cv2.imread(test_dir + 'src/' + file)
    pred = cv2.imread(
        test_dir + 'predict/' + os.path.splitext(file)[0] + '.png', 0)
    lsd = cv2.createLineSegmentDetector(0)
    # detect lines (position 0 of the returned tuple) in the image
    lines = lsd.detect(pred)[0]
    filter_lines = []
    for l in lines:
        x1, y1, x2, y2 = l.flatten()
        # discard the line segments falling in the boundary ranges of sub-images
        if abs(x2 - x1) < tol and int((x1 + x2) / 2) in cancat_range_x:
            continue
        if abs(y2 - y1) < tol and int((y1 + y2) / 2) in cancat_range_y:
            continue
        if lineMagnitude(l.flatten()) > 20:
            # let the range of line oritations be [-90,90]
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            filter_lines.append([x1, y1, x2, y2])
    # find the major orientations of line segments
    filter_lines = find_major_orientations(filter_lines)
    # merge line segments
    filter_lines = merge_lines(filter_lines)
    # remove isolated line segments (noise) and form the class boundary
    filter_lines = remove_isolated_lines(filter_lines)
    for fl in filter_lines:
        x1, y1, x2, y2 = np.rint(fl).astype(int)
        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(test_dir + 'results/' + os.path.splitext(file)[0] + '.png',
                src)
