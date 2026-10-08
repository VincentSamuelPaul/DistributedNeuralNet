import gzip, struct

def load_mnist(images_path: str, labels_path: str) -> zip:

    with gzip.open(images_path, "rb") as file:
        _, num_images, _, _ = struct.unpack(">IIII", file.read(16))
        pixel_bytes = file.read()

    with gzip.open(labels_path, "rb") as file:
        _ = struct.unpack(">II", file.read(8))
        label_bytes = file.read()

    images = []
    for i in range(num_images):
        pixels = [pixel_bytes[i*784+j] / 255.0 for j in range(784)]
        images.append(pixels)

    data = zip(images, label_bytes)

    return data