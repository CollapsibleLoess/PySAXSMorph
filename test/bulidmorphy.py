import matplotlib.pyplot as plt
import os
from datetime import datetime
import globalvars
from PIL import Image

class GaussRandField:
    def __init__(self, x, y, z, solid):
        self.position = (x, y, z)
        self.solid = solid
        self.group = -1  # 默认值为-1，表示未成组

    def set_group(self, group):
        self.group = group

    def get_group(self):
        return self.group

    def get_position(self):
        return self.position

    def is_solid(self):
        return self.solid



All = globalvars.AllParams()
def export_slices(gaussrandfield):
    exporter = Morphy(gaussrandfield)
    exporter.buildmorphy()


class Morphy:
    def __init__(self, gaussrandfield):
        self.gaussrandfield = gaussrandfield
        self.title = f"grfMorphy"
        # Get the current date and time
        now = datetime.now()
        # Format the date and time
        self.date_time = now.strftime("%Y%m%d%H")
        # Create a new folder based on date_time in output_folder
        self.new_folder = os.path.join(globalvars.output_folder, self.date_time, self.title)
        os.makedirs(self.new_folder, exist_ok=True)  # Create the new folder if it doesn't exist

    def buildmorphy(self):
        def write2DTIFFfile(gaussrandfield, x, outputfilestring):
            ymax, zmax = All.boxres, All.boxres
            image = Image.new('RGB', (ymax, zmax))
            pixels = image.load()
            for y in range(ymax):
                for z in range(zmax):
                    if gaussrandfield[x][y][z].is_solid():
                        pixels[y, z] = (0, 0, 0)  # 黑色表示固体
                    else:
                        pixels[y, z] = (255, 255, 255)  # 白色表示孔洞
            plt.savefig(os.path.join(self.new_folder, f"{outputfilestring}.tif"), dpi=600)
            plt.close()  # Close the figure to free memory
        for x in range(All.boxres):
            outputfilestring = str(x + 1).zfill(len(str(All.boxres)))
            write2DTIFFfile(self.gaussrandfield, x, outputfilestring)
