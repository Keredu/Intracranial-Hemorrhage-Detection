from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch
from torch.nn import functional as F
import os
from torchvision.models import resnet50
import cv2

def load_images(images):
    return preprocess(images)


def preprocess(images):
    inverse_norm = 255 * (0.5 * images + 0.5)
    raw_images = (inverse_norm).numpy().transpose(0, 2, 3, 1)[..., ::-1]
    return images, raw_images

def gc(model, dataset, results_dir, classes, device):
    """
    Visualize model responses given multiple images
    """
    target_layer = 'conv1'
    topk = 1
    output_dir = results_dir
    from shutil import rmtree
    if os.path.exists(output_dir): rmtree(output_dir)
    os.makedirs(output_dir)
    model.to(device)
    model.eval()

    for idx in range(len(dataset)):
        image, image_path = dataset[idx]
        image_name, ext = os.path.split(image_path)[1].split('.')
        images = torch.unsqueeze(image, 0)
        inverse_norm = 255 * (0.5 * images + 0.5)
        raw_images = (inverse_norm).numpy().transpose(0, 2, 3, 1)[..., ::-1]
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        IH = classes[probs.argmax().item()]

        image = images[0]
        raw_image = raw_images[0]
        IH_Prob = probs[0][1]

        # =====================================================================
        #model = resnet50(pretrained=True)
        target_layer = model.resnet.layer4[-1]
        #target_layer = model.layer4[-1]
        input_tensor = images# Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        target_category = 1

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category,aug_smooth=True,eigen_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        #visualization = show_cam_on_image(rgb_img, grayscale_cam)
        gcam = show_cam_on_image(raw_images[0] / 255.0, grayscale_cam)

        result_path = os.path.join(output_dir,
                                   f'{image_name}-ProbIH:{IH_Prob:.4f}-{IH}.{ext}')

        c0 = raw_image[..., 0]
        c0 = np.stack((c0, c0, c0), axis=-1)
        c1 = raw_image[..., 1]
        c1 = np.stack((c1, c1, c1), axis=-1)
        c2 = raw_image[..., 0]
        c2 = np.stack((c2, c2, c2), axis=-1)
        stack = np.concatenate((gcam, c0, c1, c2, raw_image), axis=1)


        cv2.imwrite(f'{result_path}', stack)

