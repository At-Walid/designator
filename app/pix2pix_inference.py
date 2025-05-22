import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# Import Pix2PixHD modules
from pix2pixHD.models.models import create_model
from pix2pixHD.options.test_options import TestOptions
import pix2pixHD.util.util as util
import numpy as np
import cv2
from PIL import Image

def generate_dnn_mask(image_path):
    color_map_DNN = {
        (24, 175, 120): 3,    # big rock, red
        (146, 52, 70): 1,     # bedrock, grey
        (188, 18, 5): 2,      # sand, yellow
        (187, 70, 156): 0,    # soil, beige
        (249, 79, 73): 4      # null, black
    }
    image = Image.open(image_path).convert('RGB')
    rgb_array = np.array(image)
    dnn_mask = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            rgb_value = tuple(rgb_array[i, j])
            if rgb_value in color_map_DNN:
                dnn_mask[i, j] = color_map_DNN[rgb_value]
    return dnn_mask

def generate_gan_mask(dnn_mask):
    color_map_GAN = {
        3: (255, 0, 0),       # big rock, red
        1: (128, 128, 128),   # bedrock, grey
        2: (255, 255, 0),     # sand, yellow
        0: (255, 240, 220),   # soil, beige
        4: (0, 0, 0)          # null, black
    }
    gan_mask = np.zeros((dnn_mask.shape[0], dnn_mask.shape[1], 3), dtype=np.uint8)
    for i in range(dnn_mask.shape[0]):
        for j in range(dnn_mask.shape[1]):
            label_value = dnn_mask[i, j]
            if label_value in color_map_GAN:
                gan_mask[i, j] = color_map_GAN[label_value]
    return gan_mask

opt = TestOptions().parse(save=False)
opt.nThreads = 2  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.verbose = False
opt.label_nc = 0
opt.resize_or_crop = 'none'
opt.no_instance = True
opt.isTrain = False
opt.phase = 'test'
opt.which_epoch = 'latest'
opt.use_encoded_image = False
opt.output_nc = 3
opt.onnx = None
opt.ngf = 64
opt.nef = 16
opt.instance_feat = False
opt.input_nc = 3
opt.gpu_ids = [0]
opt.data_type = 32
opt.aspect_ratio = 1.0
opt.checkpoints_dir = './checkpoints/'  # directory where your .pth files are stored
opt.name = 'ai4mars' 
# Load the model
model = create_model(opt)
model.eval()
print("Model loaded successfully.")


label_image_path = "D:/L_001.png"
dnn_mask = generate_dnn_mask(label_image_path)
label_GAN = generate_gan_mask(dnn_mask)

# Image.fromarray(dnn_mask).save("D:/DNN_image.png")
# Image.fromarray(label_GAN).save("D:/GAN_image.png")


# Save label_GAN as an image
output_gan_label_path = "D:/label_GAN.png"
Image.fromarray(label_GAN).save(output_gan_label_path)
print(f"Generated label_GAN saved to: {output_gan_label_path}")

# Load and preprocess the label image
label_img = Image.open(output_gan_label_path).convert('RGB')
transform_A = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to match Pix2PixHD input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
input_tensor = transform_A(label_img).unsqueeze(0)

# Generate the output image
with torch.no_grad():
    generated = model.inference(input_tensor, torch.tensor([0]), None)

# Convert the generated image tensor to a displayable format
output_image = util.tensor2im(generated.data[0])

# Save the generated image
output_path = "D:/G_001.png"
Image.fromarray(output_image).save(output_path)
print(f"Generated image saved to: {output_path}")

