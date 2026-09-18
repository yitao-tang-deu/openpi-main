"""Batch independent camera views without mixing their token sequences."""

import torch


def embed_camera_images(images, embed_image, *, batch_cameras: bool):
    """Return one embedding tensor per camera, in the original camera order.

    Only compatible views are combined. In particular, mixed image layouts or
    dtypes retain the original per-camera execution rather than being coerced.
    """
    if batch_cameras and len(images) > 1:
        first = images[0]
        if all(
            image.shape == first.shape and image.dtype == first.dtype and image.device == first.device
            for image in images[1:]
        ):
            # Camera-major: [camera_0's B samples, camera_1's B samples, ...].
            embeddings = embed_image(torch.cat(images, dim=0))
            return embeddings.split(first.shape[0], dim=0)
    return tuple(embed_image(image) for image in images)
