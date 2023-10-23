import numpy as np
import pandas as pd
import pathlib
import sys
import tifffile
import uuid
import xml.etree.ElementTree

def get_metamorph_channel(tiff):
    xml_str = tiff.pages[0].description
    root = xml.etree.ElementTree.fromstring(xml_str)
    channel = root.find("./PlaneInfo/prop[@id='_IllumSetting_']").attrib["value"]
    channel = channel.removesuffix("-PS").lower()
    assert channel in ("dapi", "fitc", "tritc", "cy5")
    return channel

input_path = pathlib.Path('/n/files/HiTS/lsp-data/screening-dev/DK-localization/Caitlin')
output_path = pathlib.Path(sys.argv[0]).parent / "out"
meta = pd.read_csv("IDG_DK_localization_metadata.csv")

pixel_size_um = 0.3458

directories = pd.DataFrame({"directory": meta["directory"].dropna().unique()})
directories["path"] = directories["directory"].map(
    lambda x: next((input_path / x).glob('*/*/TimePoint_1'))
)
meta = pd.merge(meta, directories, on="directory", how="left")
meta = meta[meta["path"].notna()]

meta_first = meta.groupby(['cell_line', 'tritc', 'cy5']).first().reset_index()

for t in meta_first.itertuples():
    print(t.cell_line, t.tritc, t.cy5, ":", t.well)
    input_tif_paths = sorted(t.path.glob(f"*_{t.well}_s1_w?????????-*.tif"))
    if not input_tif_paths:
        print("  (No image files found)")
        print()
        continue
    channels = {}
    for i, p in enumerate(input_tif_paths, 1):
        tiff = tifffile.TiffFile(p)
        cname = get_metamorph_channel(tiff)
        data = tiff.series[0].asarray()
        channels[cname] = data
        print(f"  {i}: {cname}")
    if t.tritc != "blank" and "tritc" not in channels:
        print("ERROR: tritc defined in metadata but no tritc image found")
        sys.exit(1)
    output_tif_path = output_path / f"{t.cell_line}_{t.tritc}_{t.cy5}.ome.tif"
    with tifffile.TiffWriter(output_tif_path) as writer:
        data = np.stack(list(channels.values()))
        marker_names = [getattr(t, k) for k in channels.keys()]
        writer.write(
            data,
            photometric="minisblack",
            resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
            compression="adobe_deflate",
            predictor=True,
            metadata={
                "UUID": uuid.uuid4().urn,
                "Channel": {"Name": marker_names},
                "PhysicalSizeX": pixel_size_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size_um,
                "PhysicalSizeYUnit": "µm",
            },
        )
    print()
