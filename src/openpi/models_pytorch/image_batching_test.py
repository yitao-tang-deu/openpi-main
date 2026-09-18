from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch.image_batching import embed_camera_images


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("channels_last", [False, True])
def test_camera_and_sample_order(batch_size, channels_last):
    torch.manual_seed(7)
    encoder = nn.Sequential(nn.Conv2d(3, 8, 2), nn.Flatten(2)).eval()
    images = [torch.randn(batch_size, 3, 4, 4) + camera for camera in range(3)]
    if channels_last:
        images = [image.permute(0, 2, 3, 1) for image in images]
    calls = []

    def encode(image):
        calls.append(image.shape[0])
        if channels_last:
            image = image.permute(0, 3, 1, 2)
        return encoder(image).transpose(1, 2)

    with torch.no_grad():
        expected = embed_camera_images(images, encode, batch_cameras=False)
        calls.clear()
        actual = embed_camera_images(images, encode, batch_cameras=True)
    assert calls == [3 * batch_size]
    assert len(actual) == 3
    for camera in range(3):
        torch.testing.assert_close(actual[camera], expected[camera])
    torch.testing.assert_close(torch.cat(actual, dim=1), torch.cat(expected, dim=1))


@pytest.mark.parametrize("mode", ["disabled", "shape", "dtype", "single", "empty"])
def test_serial_fallback(mode):
    images = [torch.ones(2, 3, 4, 4), torch.zeros(2, 3, 4, 4)]
    if mode == "shape":
        images[1] = torch.zeros(2, 3, 5, 4)
    elif mode == "dtype":
        images[1] = images[1].double()
    elif mode == "single":
        images = images[:1]
    elif mode == "empty":
        images = []
    calls = []

    def encode(image):
        calls.append(image)
        return image.flatten(2).transpose(1, 2)

    actual = embed_camera_images(images, encode, batch_cameras=mode != "disabled")
    assert len(calls) == len(images)
    for image, called, output in zip(images, calls, actual, strict=True):
        assert called is image
        torch.testing.assert_close(output, image.flatten(2).transpose(1, 2))


def test_compile_fullgraph():
    # A lightweight capture check; CUDA/Inductor performance needs a GPU run.
    def prefix(images):
        return torch.cat(embed_camera_images(images, lambda image: image.square(), batch_cameras=True), dim=1)

    images = [torch.randn(2, 5, 8) for _ in range(3)]
    compiled = torch.compile(prefix, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(images), prefix(images))


def test_prefix_masks_and_training_path():
    # Full package dependencies are available in the project's Linux environment.
    pytest.importorskip("jax")
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    class TinyTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def embed_image(self, image):
            self.calls.append(image.shape[0])
            return image.flatten(2).transpose(1, 2)

        def embed_language_tokens(self, tokens):
            return tokens[..., None].float().expand(-1, -1, 3)

    # Exercise the real prefix method without allocating the multi-billion-
    # parameter model or requiring a checkpoint.
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(pytorch_camera_batching=False)
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = TinyTower()
    model.eval()
    images = [torch.randn(2, 3, 4, 4) for _ in range(3)]
    masks = [torch.tensor([True, False]), torch.tensor([False, True]), torch.tensor([False, False])]
    tokens = torch.tensor([[1, 2], [3, 4]])
    lang_mask = torch.tensor([[True, False], [True, True]])
    expected = model.embed_prefix(images, masks, tokens, lang_mask)
    model.config.pytorch_camera_batching = True
    model.paligemma_with_expert.calls.clear()
    actual = model.embed_prefix(images, masks, tokens, lang_mask)
    assert model.paligemma_with_expert.calls == [6]
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)
    expected_mask = torch.cat([*(mask[:, None].expand(2, 16) for mask in masks), lang_mask], dim=1)
    torch.testing.assert_close(actual[1], expected_mask)
    assert not actual[2].any()

    model.train()
    model.paligemma_with_expert.calls.clear()
    training = model.embed_prefix(images, masks, tokens, lang_mask)
    assert model.paligemma_with_expert.calls == [2, 2, 2]
    for result, reference in zip(training, expected, strict=True):
        torch.testing.assert_close(result, reference)
