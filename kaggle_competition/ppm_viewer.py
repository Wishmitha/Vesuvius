from PIL import Image

with open("../../Datasets/carbon-phantom/CarbonPhantomV3.volpkg1/CarbonPhantomV3.volpkg/working/3/Col3.ppm", "rb") as file:
    raw_data = file.read()
    print(raw_data)