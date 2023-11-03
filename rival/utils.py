import torch
import cv2
import numpy as np

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def save_images(images, num_rows=1, offset_ratio=0.02, name="im.png", upper_note=None, lower_note=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    if upper_note is not None:
        h, w, c = image_.shape
        upper = 20*np.ones((70, w, c)).astype(np.uint8)
        boarder = 255*np.ones((10, w, c)).astype(np.uint8)
        cv2.putText(upper, upper_note, (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        image_ = np.concatenate([upper, boarder, image_], axis=0)
    if lower_note is not None:
        h, w, c = image_.shape
        lower = 255*np.ones((70, w, c)).astype(np.uint8)
        boarder = 255*np.ones((10, w, c)).astype(np.uint8)
        for i, note in enumerate(lower_note):
            textsize = cv2.getTextSize(note, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            textX = (512 - textsize[0]) // 2 + (i*512 + i*offset_ratio*512)
            cv2.putText(lower, note, (int(textX), 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        image_ = np.concatenate([image_, boarder, lower], axis=0)
    cv2.imwrite(name, cv2.cvtColor(image_, cv2.COLOR_RGB2BGR))