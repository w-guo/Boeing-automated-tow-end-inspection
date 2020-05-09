w, h = (968, 728)  # original image size
img_w, img_h = (256, 256)  # sub-image size

# for test image reconstruction
overlap_pct = 0  # overlap percentage between sub-images
window_size = img_w
n_w = (w - img_w * overlap_pct) // (img_w * (1 - overlap_pct)) + 1
n_h = (h - img_h * overlap_pct) // (img_h * (1 - overlap_pct)) + 1
# padded image size
pad_w = img_w * (1 - overlap_pct) * n_w + img_w * overlap_pct
pad_h = img_h * (1 - overlap_pct) * n_h + img_h * overlap_pct

aug_w = int((pad_w - w) / 2)
aug_h = int((pad_h - h) / 2)
borders = ((aug_h, aug_h), (aug_w, aug_w))