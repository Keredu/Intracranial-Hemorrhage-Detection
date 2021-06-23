import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(images):
    return preprocess(images)


def preprocess(images):
    inverse_norm = 255 * (0.5 * images + 0.5)
    raw_images = (inverse_norm).numpy().transpose(0, 2, 3, 1)[..., ::-1]
    return images, raw_images


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    c0 = raw_image[..., 0]
    c0 = np.stack((c0, c0, c0), axis=-1)
    c1 = raw_image[..., 1]
    c1 = np.stack((c1, c1, c1), axis=-1)
    c2 = raw_image[..., 0]
    c2 = np.stack((c2, c2, c2), axis=-1)
    stack = np.concatenate((gcam, c0, c1, c2, raw_image), axis=1)
    cv2.imwrite(filename, np.uint8(stack))

def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def gc_test_old(model, dataset, experiment_dir, classes, device):
    """
    Visualize model responses given multiple images
    """
    target_layer = 'layer4'
    topk = 1
    output_dir = experiment_dir
    from shutil import rmtree
    if os.path.exists(output_dir): rmtree(output_dir)
    os.makedirs(output_dir)
    model.to(device)
    model.eval()


    for idx in range(len(dataset)):
        image, image_path = dataset[idx]
        image_name = os.path.split(image_path)[1]
        images = torch.unsqueeze(image, 0)
        images, raw_images = load_images(images)
        images = images.to(device)

        bp = BackPropagation(model=model)
        probs, ids = bp.forward(images)  # sorted
        for i in range(topk):
            bp.backward(ids=ids[:, [i]])
            gradients = bp.generate()
        # Remove all the hook function in the "model"
        bp.remove_hook()
        # =====================================================================
        #print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

        gcam = GradCAM(model=model)
        _ = gcam.forward(images)

        for i in range(topk):
            # Grad-CAM
            gcam.backward(ids=ids[:, [i]])
            regions = gcam.generate(target_layer=target_layer)

            for j in range(len(images)):
                #print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
                # Grad-CAM
                result_path = osp.join(output_dir,
                                       f'{classes[ids[j, i]]}-{image_name}')
                save_gradcam(
                    filename=result_path,
                    gcam=regions[j, 0],
                    raw_image=raw_images[j],
                )


def gc_test(model, dataset, experiment_dir, classes, device):
    """
    Visualize model responses given multiple images
    """
    target_layer = 'layer4'
    topk = 1
    output_dir = experiment_dir
    from shutil import rmtree
    if os.path.exists(output_dir): rmtree(output_dir)
    os.makedirs(output_dir)
    model.to(device)
    model.eval()

    for idx in range(len(dataset)):
        image, image_path = dataset[idx]
        image_name = os.path.split(image_path)[1]
        images = torch.unsqueeze(image, 0)
        images, raw_images = load_images(images)
        images = images.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        # =====================================================================
        #print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

        gcam = GradCAM(model=model)
        _ = gcam.forward(images)

        # Grad-CAM
        gcam.backward(ids=torch.Tensor([[1]]).long().to(device)) # IH class
        regions = gcam.generate(target_layer=target_layer)

        # Grad-CAM
        image_name, ext = image_name.split('.')
        result_name = f'{image_name}-{probs[0,1]:.4f}.{ext}'
        result_path = osp.join(output_dir, result_name)
        save_gradcam(
            filename=result_path,
            gcam=regions[0, 0],
            raw_image=raw_images[0],
        )
