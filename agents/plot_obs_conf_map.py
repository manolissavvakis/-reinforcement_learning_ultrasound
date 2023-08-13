import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def plot_confidence_map(bmode, map, step, save_dir=None):

    if isinstance(bmode, str):
        bmode = np.asarray(Image.open(bmode))
    if isinstance(map, str):
        map = np.asarray(Image.open(map))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(bmode, cmap='gray')
    ax1.set_title('Bmode')
    ax1.axis('off')

    ax2.imshow(map, cmap='gray')
    ax2.set_title('Confidence Map')
    ax2.axis('off')

    plt.show()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'step_%s_conf_map.png' %str(step)))