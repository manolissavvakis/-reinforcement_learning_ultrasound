import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_confidence_map(bmode, map, save_mode):

    if isinstance(bmode, str) and isinstance(map,str):
        bmode = Image.open(bmode)
        map = Image.open(map)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(bmode)
    ax1.set_title('Observation')
    ax1.axis('off')

    ax2.imshow(map, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Confidence Map')
    ax2.axis('off')

    plt.show()

    if save_mode:
        plt.savefig('obs_conf_map.png')