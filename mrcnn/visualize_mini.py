import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import scipy.ndimage
import skimage
import skimage.measure
import Mask_RCNN.visualize as visualize

def _inrange(x, low, high):
    return np.logical_and(x >= low, x < high)

def clamp(x): 
  return max(0, min(x, 255))

def rgb2hex(r,g,b):
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(int(255*r)), clamp(int(255*g)), clamp(int(255*b)))

def apply_minimask(image, minimask, bbox4, color, alpha=0.5):
    """Apply the given masks to the image; modifies the image data directly directly
    Parameters:
    ===========
    image: np.array(shape=[H, W, C])
    minimask: np.array(shape=[h, w])
    bbox4: [y1, x1, y2, x2]
    color: np.array(shape=[3,])
        RGB taking values between 0 and 1
    alpha: float
        ranges from 0 to 1. when alpha is 0, effectively do nothing

    Returns:
    ========
    image: np.ndarray
    image with minimasks applied. The application of the minimask linearly interpolates the minimask onto the 
        whole canvas with factor alpha
    """
    H, W, C = image.shape
    h, w = minimask.shape
    y1, x1, y2, x2 = np.round(bbox4).astype(np.int)
    assert x2 >= x1
    assert y2 >= y1
    _box4 = np.round(np.clip(bbox4, 0, np.array([H-1, W-1, H-1, W-1]))).astype(np.int) # clip to image
    _y1, _x1, _y2, _x2 = _box4
    if _y2 - _y1 <= 0  or _x2 - _x1 <= 0:
        return image # check that minimask has area within image
    
    minimask_preimg = scipy.misc.imresize(
        minimask.astype(np.float32), (y2 - y1, x2 - x1), interp='bilinear') / 255.0
    threshold = 0.5
    minimask_preimg = np.where(minimask_preimg >= threshold, 1, 0)

    mmy1 = _y1 - y1 # what if part of the minimask is outside of the image -- several cases
    mmx1 = _x1 - x1
    mmy2 = mmy1 + _y2 - _y1
    mmx2 = mmx1 + _x2 - _x1
    amm_preimg = minimask_preimg[mmy1:mmy2, mmx1:mmx2, np.newaxis] * alpha  # alpha * minimask within image
    color = np.array(color).reshape([1, 1, 3])
    image[_y1:_y2, _x1:_x2, :] = image[_y1:_y2, _x1:_x2, :] * (1 - amm_preimg) +\
                             (color * 255           * amm_preimg)
    return image


def display_minimasks(image, boxes, minimasks, class_ids, class_names,
                      scores=None, title=None, draw_boxes=True, mask_alpha=.5,
                      figsize=(16, 16), ax=None, colors=None, mask_thresh=None,
                      color_by_class=False):
    """
    Displays minimasks onto an image, and plot it.

    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    minimasks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.

    """
    # Number of instances
    mmh, mmw, N = minimasks.shape
    if not N:
        pass
        #print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == minimasks.shape[-1] == class_ids.shape[0]


    if not ax :
        _, ax = plt.subplots(1, figsize=figsize)
    

    # Generate random colors
    if colors is None:
        if color_by_class:
            colors = visualize.random_colors(1 + np.max(class_ids))
        else:
            colors = visualize.random_colors(N)
    else:
        if len(colors) != N:
            raise ValueError("colors has length %d should be %d" % (len(colors), N))

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    if mask_thresh is None:
        mask_thresh = (minimasks.max() + minimasks.min()) / 2.
    minimasks = minimasks >= mask_thresh
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[class_ids[i]] if color_by_class else colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.

            continue
        y1, x1, y2, x2 = boxes[i].astype(np.int)
        if draw_boxes:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = None if scores is None else scores[i]
        label = "" if class_names is None else class_names[class_id]
        caption = "{} {:.2f}".format(label, score) if score else label
        if class_names is not None or score is not None:
            _x = random.randint(x1, (x1 + x2) // 2)
            _y = random.randint(y1, (y1 + y2) // 2)            
            ax.text(_x, _y + 8, caption,
                    color='w', size=11, backgroundcolor="none")

        # Mask
        minimask = minimasks[:, :, i]
        bbox = boxes[i, :4]
        if mask_alpha > 0:
            masked_image = apply_minimask(masked_image, minimask, bbox, color, alpha=mask_alpha)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_minimask = np.zeros(
            (minimask.shape[0] + 2, minimask.shape[1] + 2), dtype=np.uint8)
        padded_minimask[1:-1, 1:-1] = minimask
        minimask_contours = skimage.measure.find_contours(padded_minimask, 0.5)
        for pmmverts in minimask_contours:
            # Subtract the padding and flip (y, x) to (x, y)
            mmxyverts = np.fliplr(pmmverts) - 1
            mmxyscale = np.array([(x2 - x1) / float(mmw), (y2 - y1) / float(mmh)]).reshape([1, 2])
            verts = np.array([x1, y1]) + mmxyverts * mmxyscale
            p = Polygon(verts, facecolor="none", edgecolor=color, linewidth=2)
            ax.add_patch(p)
                
    ax.imshow(masked_image.astype(np.uint8))

