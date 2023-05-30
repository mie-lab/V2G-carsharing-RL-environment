import matplotlib.pyplot as plt
from IPython import display
import matplotlib as mpl
from matplotlib import cm

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def __call__(self, val):
        return self.scalarMap.to_rgba(val)
    
my_cmap = MplColorHelper("viridis", 0, 1)

def show_soc(img):
    plt.figure(figsize=(8,8))
    plt.clf()
    plt.imshow(img)
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())