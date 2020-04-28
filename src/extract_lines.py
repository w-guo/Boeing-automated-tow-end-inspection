import fnmatch
import cv2
import os
from utils import lineMagnitude, find_major_orientations, merge_lines, remove_isolated_lines
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\BARC\\Desktop\\AFP\\binary_seg')


h, w = (728, 968)
img_h, img_w = (256, 256)

pad_h = (h//img_h+1)*img_h
pad_w = (w//img_w+1)*img_w

aug_h = (pad_h-h)/2
aug_w = (pad_w-w)/2

tol = 7
cancat_range_x = []
cancat_range_y = []
for i in range(img_w, pad_w, img_w):
    subimg_bd_x = int(i-aug_w)
    range_x = range(subimg_bd_x-tol, subimg_bd_x+tol)
    cancat_range_x += list(range_x)

for j in range(img_h, pad_h, img_h):
    subimg_bd_y = int(j-aug_h)
    range_y = range(subimg_bd_y-tol, subimg_bd_y+tol)
    cancat_range_y += list(range_y)

subfolder = 'unet_xception_resnet_nsgd32_lovasz/'
test_imgs = fnmatch.filter(os.listdir('./test/src/'), '*.jpg')

for filename in test_imgs:
    src = cv2.imread('./test/src/' + filename)
    pred = cv2.imread('./test/results/' + subfolder +
                      os.path.splitext(filename)[0]+'.png', 0)
    lsd = cv2.createLineSegmentDetector(0)
    # Detect lines in the image
    # Position 0 of the returned tuple are the detected lines
    lines = lsd.detect(pred)[0]
    filter_lines = []
    for l in lines:
        x1, y1, x2, y2 = l.flatten()
        # discard the line segments at the boundary of sub-images
        if abs(x2-x1) < 7 and int((x1+x2)/2) in cancat_range_x:
            continue
        if abs(y2-y1) < 7 and int((y1+y2)/2) in cancat_range_y:
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
    cv2.imwrite('./test/line_extraction/' +
                os.path.splitext(filename)[0]+'_fo_ml_rm.png', src)
