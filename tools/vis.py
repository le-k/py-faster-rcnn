import sys
import cv2
import matplotlib.pyplot as plt

def get_classes():
    cate_table_file = "/root/packages/py-faster-rcnn/tools/cat_table_purged.txt"
    all_classes = ['__background__']
    with open(cate_table_file, 'r') as fh:
        for line in fh:
            vals = line.strip().split('\t')
            all_classes.append(vals[-1])
    return all_classes

CLASSES = get_classes()
    
def vis_detections(imsave_file, im, x, y, w, h, c1, c1s):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    ct = 0
    for i in range(len(x)):
        ct += 1
        score = c1s[i]
        ax.add_patch(
            plt.Rectangle( (x[i], y[i]), w[i], h[i], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(x[i], y[i] - 2,
                '{:d}. {:s} {:.3f}'.format(ct, CLASSES[c1[i]], c1s[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                 'confidence = {:.2f}').format(ct, 0.2), fontsize=14)
    print('{} detections'.format(ct))

    plt.axis('off')
    plt.tight_layout()
    print "xx", imsave_file
    plt.savefig(imsave_file)
    plt.close()

x = []
y = []
w = []
h = []
c1 = []
c1s = []
with open(sys.argv[1], 'r') as fh:
    for line in fh:
        line = line.strip()
        vals = line.split()
        if vals[0] == "count":
            if len(x) > 0:
                im = cv2.imread(load_file)
                vis_detections(save_file, im, x, y, w, h, c1, c1s)
                x = []
                y = []
                w = []
                h = []
                c1 = []
                c1s = []
        else:
            load_file = vals[0]
            f1 = load_file.split("_")
            print f1, f1[0]
            save_file = "/drive3/tmp/tmp_" + f1[2]
            x.append(int(vals[1]))
            y.append(int(vals[2]))
            w.append(int(vals[3]))
            h.append(int(vals[4]))
            c1.append(int(vals[5]))
            c1s.append(float(vals[6]))
    
