import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import flow_to_image

def visualize_flow(flow):
    flow_img = flow_to_image(flow).permute(1,2,0).cpu().numpy()
    plt.imshow(flow_img)
    plt.show()

def visualize_image(img):
    plt.imshow(img.permute(1,2,0).cpu().numpy())
    plt.show()
