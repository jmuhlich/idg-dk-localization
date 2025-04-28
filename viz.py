import napari
import numpy as np
import pathlib
import sys
import tifffile

plate, row = sys.argv[1:]
base = pathlib.Path(__file__).resolve().parent / "in" / "viewers" / "images" / f"plate{plate}"

well_width = 4 if plate in ("14", "20", "24") else 3
width = well_width * 8 * 2048
height = 3 * 2048
img = np.zeros((4, height, width), "uint16")

for col, colname in enumerate([2,3,6,7,8,9,10,11]):
    for field, fieldname in enumerate(range(1, well_width * 3 + 1)):
        for ch, chname in enumerate(range(1, 4 + 1)):
            x = (col * well_width + field % well_width) * 2048
            y = field // 3 * 2048
            pattern = f"*_{row}{colname:02}_s{fieldname}_w{chname}*.tif"
            try:
                path = next(base.glob(pattern))
            except StopIteration:
                print(f"Couldn't load: {pattern}")
                continue
            img[ch, y:y+2048, x:x+2048] = tifffile.imread(path)

colors = ("gray", "red", "green", "bop blue")
viewer = napari.Viewer()
for c, i in zip(colors, img):
    viewer.add_image(
        [i, i[::4,::4], i[::16,::16]],
        contrast_limits=(0, 65535),
        colormap=c,
        blending="additive",
    )
napari.run()
