import os
import csv
import time
import math
import random
import shutil
import subprocess
import numpy as np
from collections import OrderedDict
from PIL import Image

import cv2
import psutil
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.utils import save_image
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex


import airsim
from airsim.types import Pose

import pymoo
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.algorithms.so_genetic_algorithm import comp_by_cv_and_fitness, FitnessSurvival
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
import pymoo.problems.multi
from pymoo.util.dominator import Dominator

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA as SGA
from pymoo.visualization.scatter import Scatter

from torch.autograd import Variable
from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer
from pix2pixHD.util import html
from pix2pixHD.data.base_dataset import BaseDataset, get_params, get_transform, normalize

from torchvision.utils import save_image
import torchvision

from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import euclidean_distances
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex
import pytorch_lightning as pl

# Set up GPU context for mxnet
# ctx = mx.gpu(0)

# Initialize TestOptions
opt = TestOptions().parse(save=False)
opt.nThreads = 2
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
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

visualizer = Visualizer(opt)
web_dir = os.path.join('./results/Random/')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

opt.checkpoints_dir = 'app/checkpoints/' 
opt.name = 'ai4mars' 
def preprocess_image(image_path: str, return_tensor: bool = False):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        image_normalized = np.asarray(image, dtype=np.float32) / 255.0
        
        if return_tensor:
            image_normalized = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
            return image, image_tensor
        
        return image
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
def generate_rgb_mask(dnn_mask):
    color_map_RGB = {
        3: (24, 175, 120),    # big rock, red
        1: (146, 52, 70),     # bedrock, grey
        2: (188, 18, 5),      # sand, yellow
        0: (187, 70, 156),    # soil, beige
        4: (249, 79, 73)      # null, black
    }

    rgb_array = np.zeros((dnn_mask.shape[0], dnn_mask.shape[1], 3), dtype=np.uint8)

    for label, rgb in color_map_RGB.items():
        rgb_array[dnn_mask == label] = rgb  # Apply corresponding RGB color

    # Convert to an image and save
    # rgb_image = Image.fromarray(rgb_array, 'RGB')
    # rgb_image.save(output_path)
    return rgb_array
def compute_all_color_proportion(image_path):
        # Load the image with PIL
        img = Image.open(image_path)

        # Convert the image to a numpy array
        img_np = np.array(img)

        # Ensure the image has 3 channels (RGB)
        if len(img_np.shape) == 2:  # Grayscale image
            img_np = np.stack((img_np,) * 3, axis=-1)
        elif img_np.shape[2] == 4:  # RGBA image, convert to RGB
            img_np = img_np[:, :, :3]

        # Flatten the image to a 1D array
        flattened_img = img_np.reshape((-1, 3))

        # Get unique colors and their counts
        unique_colors, counts = np.unique(flattened_img, axis=0, return_counts=True)

        # Total number of pixels in the image
        total_pixels = img_np.shape[0] * img_np.shape[1]  # Assuming 2D image

        # Compute the proportions for each unique color
        proportions = counts / total_pixels

        # Check if any color has a proportion higher than 70%
        for color, proportion in zip(unique_colors, proportions):
            if proportion > 0.60:
                return True, color.tolist(), proportion

        return False, None, None
class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # self.model_weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, y), on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the saved model

model = ImageSegmentationModel()
model.load_state_dict(torch.load("app/dnn_model/retrained_model_checkpoint.pth"))
model.eval()
model.to(device)
def predict_img(f):
        test_image, test_image_tensor = preprocess_image(f, return_tensor=True)
        test_image_tensor = test_image_tensor.to(device)

        # Perform prediction
        with torch.no_grad():
            prediction = model(test_image_tensor)
            predicted_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        # Load the ground truth segmentation
        

        return predicted_mask
def compute_iou(mask1, mask2, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(mask1 == cls, mask2 == cls).sum()
        union = np.logical_or(mask1 == cls, mask2 == cls).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def compute_miou(ious):
    # Filter out 'nan' values and compute the mean
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if len(valid_ious) == 0:
        return float('nan')
    else:
        return np.mean(valid_ious)      
class GaGan:
    def __init__(self):
        self.server = None
        self.vehicle_name = None
        self.client = airsim.CarClient()
        self.carControls = airsim.CarControls()
        self.dir = "./results/Random/"
    def collect_data_to_csv(self, total=10000, freq=0.1):
        road_pos = []
        self.client.confirmConnection()
        
        # Set the camera pose
        camera_position = airsim.Vector3r(0, 0, -1)  # x, y, z coordinates above the car
        camera_orientation = airsim.to_quaternion(-0.5, 0, 0)  # roll, pitch (downward), yaw in radians
        self.client.simSetCameraPose("0", airsim.Pose(camera_position, camera_orientation))
        self.client.simSetCameraPose("1", airsim.Pose(camera_position, camera_orientation))


        for i in range(total):

            # Generate random positions within the landscape boundaries with z = 0
            x_pos = random.uniform(-430, 560)  # Adjust the range as per your landscape dimensions
            y_pos = random.uniform(-540, 430)  # Adjust the range as per your landscape dimensions
            z_pos = 0  # Fixed z position

            # Generate random orientations
            w_ori = random.uniform(0, 1)
            x_ori = random.uniform(-1, 1) / 1000
            y_ori = random.uniform(-1, 1) / 1000
            z_ori = 0 #random.uniform(-1, 1)

            # Set the vehicle pose to these random values
            # print(x_pos, y_pos, z_pos)
            position = airsim.Vector3r(x_pos, y_pos, z_pos)
            orientation = airsim.Quaternionr(w_val=w_ori, x_val=x_ori, y_val=y_ori, z_val=z_ori)
            pose = Pose(position_val=position, orientation_val=orientation)
            self.GaGan.client.simSetCameraPose("0", pos)
            self.GaGan.client.simSetCameraPose("1", pos)
            time.sleep(1)

            # Capture the position
            pos = self.client.simGetVehiclePose(self.vehicle_name)

            # Convert Pose data into dictionary form for easier CSV writing
            pos_dict = {
                'x_pos': pos.position.x_val,
                'y_pos': pos.position.y_val,
                'z_pos': pos.position.z_val,
                'w_ori': pos.orientation.w_val,
                'x_ori': pos.orientation.x_val,
                'y_ori': pos.orientation.y_val,
                'z_ori': pos.orientation.z_val,
                'p_r_y': airsim.to_eularian_angles(pos.orientation)
            }
            print('x_pos', pos.position.x_val,
                'y_pos', pos.position.y_val,
                'z_pos', pos.position.z_val,
                'w_ori', pos.orientation.w_val,
                'x_ori', pos.orientation.x_val,
                'y_ori', pos.orientation.y_val,
                'z_ori', pos.orientation.z_val,
                'p_r_y', airsim.to_eularian_angles(pos.orientation))
            road_pos.append(pos_dict)
            # time.sleep(freq)

            # Capture images
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),  # label
                airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)  # scene vision image in png format
            ])

            # Save images to ./results/random/airsim directory
            save_path = "./results/Random/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for response_idx, response in enumerate(responses):
                try:
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    
                    # Debugging information
                    print(f"Response Index: {response_idx}, Image Shape: {img_rgb.shape}, Data Type: {img_rgb.dtype}")
                    
                    if response_idx == 0:  # Segmentation mask
                        label_filename = os.path.join(save_path, f"L_{i}.png")
                        try:
                            cv2.imwrite(label_filename, img_rgb)
                        except cv2.error as e:
                            print(f"Failed to write segmentation mask image: {e}")
                    elif response_idx == 1:  # Scene image
                        scene_filename = os.path.join(save_path, f"S_{i}.png")
                        try:
                            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # Convert to greyscale
                            cv2.imwrite(scene_filename, img_gray)
                        except cv2.error as e:
                            print(f"Failed to write scene image: {e}")
                except Exception as e:
                    print(f"Error processing image at index {response_idx}: {e}")
        # Write data to CSV file
        with open('app/road_positions.csv', 'w', newline='') as csvfile:
            fieldnames = ['x_pos', 'y_pos', 'z_pos', 'w_ori', 'x_ori', 'y_ori', 'z_ori', 'p_r_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pos in road_pos:
                writer.writerow(pos)

        print('Collected ' + str(len(road_pos)) + ' points')

    # def is_car_stationary(self):
    #     # Get the car's velocity
    #     car_state = self.client.getCarState(self.vehicle_name)
    #     car_velocity = car_state.kinematics_estimated.linear_velocity

    #     # Check if the car's velocity is near zero
    #     velocity_threshold = 0.1  # Adjust this threshold based on your scenario
    #     is_stationary = (abs(car_velocity.x_val) < velocity_threshold and
    #                      abs(car_velocity.y_val) < velocity_threshold and
    #                      abs(car_velocity.z_val) < velocity_threshold)

        return is_stationary

    def searchAlgo(self, numSet):
        path = "./results/Random/"
        self.dir = path
        os.makedirs(path, exist_ok = True) 
        with open('app/road_positions.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]

        GAN = getGAN()
        archive2 = []
        archive_features = []
        type_distance = 'FeatureDistance'
        threshold_diversity = 30
        self.archive = []
        for c in range(100):
            X = [[random.uniform(0, 1), random.uniform(0, 1)] for _ in range(12)]
            print('@@@@@@@@@@@@@@@@', c)
            for x in X:
                row = rows[int(x[0] * len(rows))]
                pos_dict = {
                    'x_pos': float(row['x_pos']), 'y_pos': float(row['y_pos']), 'z_pos': float(row['z_pos']),
                    'w_ori': float(row['w_ori']), 'x_ori': float(row['x_ori']), 'y_ori': float(row['y_ori']),
                    'z_ori': float(row['z_ori']), 'p_r_y': list(row['p_r_y'])
                }
                position = airsim.Vector3r(x_val=pos_dict['x_pos'], y_val=pos_dict['y_pos'], z_val=pos_dict['z_pos'])
                orientation = airsim.Quaternionr(w_val=x[1], x_val=pos_dict['x_ori'], y_val=pos_dict['y_ori'],
                                                 z_val=pos_dict['z_ori'])
                pos = Pose(position_val=position, orientation_val=orientation)
                self.client.simSetCameraPose("0", pos)
                self.client.simSetCameraPose("1", pos)


                start_time = time.time()
                # while time.time() - start_time > 10 and not self.is_car_stationary():
                #     print("Waiting for the car to settle...")
                #     time.sleep(1)  # Check every second

                # if not self.is_car_stationary():
                #     self.client.setCarControls(airsim.CarControls(brake=1.0))
                #     print("Applying brakes...")
                #     time.sleep(2)  # Apply brakes for 5 seconds
                #     self.client.setCarControls(airsim.CarControls(brake=0.0))
                
                save_path = os.path.join(path, "R_" + str(x[0]) + "_" + str(x[1]) + ".png")
                simulated_path = os.path.join(path, "S_" + str(x[0]) + "_" + str(x[1]) + ".png")
                label_path = os.path.join(path, "L_" + str(x[0]) + "_" + str(x[1]) + ".png")
                
                if not os.path.isfile(save_path):
                    if self.client.simGetCollisionInfo().has_collided:
                        self.reset()
                        time.sleep(1)
                    else:
                        img_path = self.retrieveImages(x)
                        

                        sim_img = cv2.imread(simulated_path)
                        img_gray = cv2.cvtColor(sim_img, cv2.COLOR_RGB2GRAY)
                        cv2.imwrite(simulated_path, img_gray)
                        t4 = time.time()
                        label_DNN = generate_dnn_mask(label_path)
                        # Image.fromarray(label_DNN.astype(np.uint8)).save(os.path.join("D:/results/airsim", "D_" + str(x[0]) + "_" + str(x[1]) + ".png"))
                        label_GAN = generate_gan_mask(label_DNN)


                        img = Image.fromarray((label_GAN * 1).astype(np.uint8))
                        img.save(os.path.join(path, "G_" + str(x[0]) + "_" + str(x[1]) + ".png"))

                        t6 = time.time()
                        # p.suspend()
                        transform_A = get_transform(opt, get_params(opt, img.size))
                        img_A = transform_A(img.convert('RGB')).unsqueeze(0)
                        generated = GAN.inference(img_A, torch.tensor([0]), None)
                        util.save_image(util.tensor2im(generated.data[0]), save_path)
                        proportion = compute_all_color_proportion(label_path)

                        pred = predict_img(save_path) 
                        f = compute_miou(compute_iou(pred, label_DNN, 4))

                        pred_path = os.path.join(path, "P_" + str(x[0]) + "_" + str(x[1]) + ".png")
                        rgb_pred = Image.fromarray(generate_rgb_mask(pred), 'RGB')
                        rgb_pred.save(pred_path)


                        if(not proportion[0]):
                            archive2.append(save_path)
                        self.archive.append({'img': save_path, 'individual': (x[0], x[1]), 'entry': 0, 'rot': int(x[1]*360), 'F': (f, random.uniform(1.0, 30.0)), 'PixAcc': 0})

                with open(path + '/archive.txt', 'w') as fp:
                    for item in self.archive:
                        fp.write("%s\n" % item)


    def begin_server(self, server, vehicle_name):
        self.server = server
        self.vehicle_name = vehicle_name
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.simEnableWeather(True)
            self.client.reset()
        except:
            try:
                subprocess.Popen(self.server)
                time.sleep(10)
                self.client = airsim.CarClient()
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.simEnableWeather(True)
                # self.assignIDs()
            except FileNotFoundError:
                print("Airsim not found: " + self.server)
            except Exception as e:
                print("Error occurred while starting Airsim: " + str(e))
        print("Airsim started successfully!")

    def retrieveImages(self, x):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),  # label
            airsim.ImageRequest("1", airsim.ImageType.Scene),  # scene vision image in png format
        ])
        files = []
        labels = ['L', 'S']
        for response_idx, response in enumerate(responses):
            filename = os.path.join(self.dir, labels[response_idx] + '_' + str(x[0]) + "_" + str(x[1]))
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress:
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else:
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)
            files.append(os.path.normpath(filename + '.png'))
        return files

    def reset(self):
        car_controls = airsim.CarControls()
        car_controls.throttle = 0
        car_controls.steering = 0
        car_controls.speed = 0
        self.client.setCarControls(car_controls)
        self.client.reset()

def mask_label(file):
    airsim_to_cityscapes = {
        (187, 70, 156): (128, 64, 128),
        (112, 105, 191): (244, 35, 232),
        (89, 121, 72):(70, 70, 70),
        (0,53,65):(70, 70, 70),
        (28,34,108):(70, 70, 70),
        (49,89,160):(70, 70, 70),
        (190, 225, 64):(102, 102, 156),
        (206, 190, 59):(190, 153, 153),
        (135, 169, 180):(153, 153, 153),
        (115, 176, 195):(244, 35, 232),
        (81, 13, 36):(107, 142, 35),
        (29, 26, 199):(70, 130, 180),
        (102, 16, 239):(70, 130, 180),
        (189, 135, 188):(220, 20, 60),
        (156, 198, 23):(255, 0, 0),
        (161, 171, 27):(0, 0, 142),
        (68, 218, 116):(0, 0, 70),
        (11, 236, 9):(0, 60, 100),
        (196, 30, 8):(0, 80, 100),
        (121, 67, 28):(0, 0, 230),
        (148,66,130):(0, 0, 142),
        (250,170,30):(244, 35, 232),
        (153,108,6): (128, 64, 128),
        (131,182,184): (153, 153, 153),
        (0, 0, 0): (70, 130, 180)
    }

    airsim_rgb_mask = np.array(Image.open(file))
    airsim_rgb_mask = cv2.cvtColor(airsim_rgb_mask, cv2.COLOR_BGRA2BGR)
    lookup_table = {str(k): v for k, v in airsim_to_cityscapes.items()}

    def convert_to_cityscapes_rgb(rgb_mask):
        height, width, _ = rgb_mask.shape
        reshaped_mask = rgb_mask.reshape(-1, 3)
        mask_str = np.array([str(tuple(pixel)) for pixel in reshaped_mask])
        cityscapes_colors = np.array([lookup_table.get(color_str, [0, 0, 0]) for color_str in mask_str])
        return cityscapes_colors.reshape(height, width, 3)

    return convert_to_cityscapes_rgb(airsim_rgb_mask)

def getGAN():
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
        model.opt = opt
    else:
        model = None
    print('Model loaded')
    return model

class RankAndCrowdingSurvival(Survival):
    def __init__(self):
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, n_survive, D=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        survivors = []
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            crowding_of_front = calc_crowding_distance(F[front, :])
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]
            else:
                I = np.arange(len(front))
            survivors.extend(front[I])
        return pop[survivors]

def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)
    else:
        is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0] if filter_out_duplicates else np.arange(n_points)
        _F = F[is_unique]
        I = np.argsort(_F, axis=0, kind='mergesort')
        _F = _F[I, np.arange(n_obj)]
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist[:-1] / norm, dist[1:] / norm
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd
    return crowding

def find_duplicates(X, epsilon):
    D = np.linalg.norm(X[:, None] - X, axis=2)
    return np.any((D <= epsilon) & (D > 0), axis=1)

def randomized_argsort(a, order='ascending', method='numpy'):
    if method == 'numpy':
        I = np.argsort(a, kind='mergesort')
    else:
        raise ValueError(f"Unknown method {method}.")
    return I if order == 'ascending' else I[::-1]
