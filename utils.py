import math
import os
from enum import Enum

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset
from torchinfo import summary
from torchvision.io import read_image
from torchvision.models import resnet50
from torchvision.transforms import transforms

from models.CoAtNet import CoAtNet
from models.ViT import ViT

R = 6371.0


class ModelType(Enum):
    CNN = 1
    ViT = 2
    CoAtNet = 3


image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

max_img_per_state = -1


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, dataset_type, split_ratio=0.9, use_county=False):
        self.img_dir = img_dir
        self.use_county = use_county
        self.dataset_type = dataset_type
        self.index = dict()
        self.county_index = dict()
        idx = 0
        for subdir in os.listdir(self.img_dir):
            subdir_path = os.path.join(self.img_dir, subdir)
            index_file_name = os.path.join(subdir_path, "info.txt")
            index_file = open(index_file_name, "r")
            lines = index_file.readlines()
            for i in range(len(lines) // 2):
                # split data into train and validation uniformly
                item_in_train = False
                if idx % (1 / split_ratio) < 1:
                    item_in_train = True
                if dataset_type == "train" and item_in_train \
                        or dataset_type == "val" and not item_in_train \
                        or dataset_type == "test":
                    # add item to dataset
                    key = subdir + "/" + lines[i * 2].strip().split("//")[-1]
                    coordinates = lines[i * 2 + 1].strip().split(" ")
                    latitudes = float(coordinates[0]) / 180 + 0.5  # rescale to [0, 1]
                    longitudes = float(coordinates[1]) / 360 + 0.5  # rescale to [0, 1]
                    self.index[key] = (latitudes, longitudes)
                    self.county_index[key] = subdir
                idx += 1
                if i + 1 == max_img_per_state:
                    break
            index_file.close()
        self.keys = list(self.index.keys())
        print("Created images index with", len(self.keys), "images")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.keys[idx])
        image = read_image(img_path) / 255.0
        image = image_transform(image)
        coordinates = self.index[self.keys[idx]]
        label = torch.FloatTensor(coordinates)
        if self.use_county:
            return image, label, self.keys[idx].split("/", 2)[-1], self.county_index[self.keys[idx]]
        else:
            return image, label


def get_model(model_type: ModelType) -> torch.nn.Module:
    model = None

    if model_type == ModelType.CNN:
        model = resnet50(num_classes=2)
    elif model_type == ModelType.ViT:
        # ViT-L/16 model type
        model = ViT(image_size=256, patch_size=16, num_layers=24, num_heads=16, hidden_dim=1024,
                    mlp_dim=4096, dropout=0.2, attention_dropout=0.2, num_classes=2)
    elif model_type == ModelType.CoAtNet:
        # CoAtNet4 model type
        num_blocks = [2, 2, 12, 28, 2]  # L
        channels = [192, 192, 384, 768, 1536]  # D
        model = CoAtNet((256, 256), 3, num_blocks, channels, num_classes=2)

    if os.path.isfile("./trained_models/" + str(model_type.name) + ".pth"):
        model.load_state_dict(torch.load("./trained_models/" + str(model_type.name) + ".pth"))
        print(os.path.curdir)
        print("Loaded model state from " + str(model_type.name) + ".pth")
    else:
        print("No model state found for " + str(model_type.name) + ".pth")
        print("Untrained model will be used")
    summary(model, (1, 3, 256, 256), depth=10)
    print(model.eval())
    return model


def to_degrees(input):
    input[0] = (input[0] % 1 - 0.5) * 180
    input[1] = (input[1] % 1 - 0.5) * 360
    return input


def avg_dist(predictions: np.ndarray, actuals: np.ndarray):
    s = 0
    for i in range(len(predictions)):
        lat1 = (predictions[i][0] - 0.5) * math.pi
        lon1 = (predictions[i][1] - 0.5) * 2 * math.pi
        lat2 = (actuals[i][0] - 0.5) * math.pi
        lon2 = (actuals[i][1] - 0.5) * 2 * math.pi
        dist = math.acos(
            math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * R
        s += dist
    return s / len(predictions)


class HaversineLoss(_Loss):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, pred, actual):
        pred_lat, pred_lon = pred[:, 0], pred[:, 1]
        actual_lat, actual_lon = actual[:, 0], actual[:, 1]
        distance = self.haversine_distance(pred_lat, pred_lon, actual_lat, actual_lon)
        return torch.mean(distance)

    def haversine_distance(self, lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor,
                           lon2: torch.Tensor) -> torch.Tensor:
        # Convert latitude and longitude from degrees to radians
        lat1_rad = (lat1 - 0.5) * math.pi
        lon1_rad = (lon1 - 0.5) * 2 * math.pi
        lat2_rad = (lat2 - 0.5) * math.pi
        lon2_rad = (lon2 - 0.5) * 2 * math.pi

        # Calculate differences between latitudes and longitudes
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Apply Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        # Earth's radius in km (mean radius)

        return c * R


def ss_total(data: CustomImageDataset):
    print("Computing sum of squares residuals for dataset")
    labels = torch.tensor(list(data.index.values()))
    mean = torch.mean(labels, dim=0)
    ss_tot = torch.sum((labels - mean) ** 2)
    print("Total sum of squares:", ss_tot.item())
    return ss_tot


class ModelEvaluator:

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.train_ss_total = ss_total(self.train_dataset)
        self.train_num_samples = len(self.train_dataset)
        self.train_mae = []
        self.train_rmse = []
        self.train_r2 = []
        self.train_mean_dist = []

        self.val_dataset = val_dataset
        self.val_ss_total = ss_total(self.val_dataset)
        self.val_num_samples = len(self.val_dataset)
        self.val_mae = []
        self.val_rmse = []
        self.val_r2 = []
        self.val_mean_dist = []

    def compute_metrics(self, predictons: np.ndarray, actuals: np.ndarray, dataset_type: str):
        # MAE
        mae = np.mean(np.abs(predictons - actuals))
        # RMSE
        rmse = np.sqrt(np.mean(np.square(predictons - actuals)))
        # R squared
        if dataset_type == "train":
            num_samples = self.train_num_samples
            ss_total = self.train_ss_total
        else:
            num_samples = self.val_num_samples
            ss_total = self.val_ss_total
        ss_reg = np.sum(np.square(actuals - predictons)) * num_samples / len(predictons)
        r2 = 1 - (ss_reg / ss_total)
        # Mean geographic distance error
        mean_dist = avg_dist(predictons, actuals)

        return mae, rmse, r2, mean_dist

    def plot_metrics(self):
        # Plot MAE
        plt.plot(self.train_mae, label="train", color="blue")
        plt.plot(self.val_mae, label="val", color="red")
        plt.legend()
        plt.title("MAE")
        plt.savefig("plots/MAE.png")
        plt.show()

        # Plot RMSE
        plt.plot(self.train_rmse, label="train", color="blue")
        plt.plot(self.val_rmse, label="val", color="red")
        plt.legend()
        plt.title("RMSE")
        plt.savefig("plots/RMSE.png")
        plt.show()

        # Plot R^2
        plt.plot(self.train_r2, label="train", color="blue")
        plt.plot(self.val_r2, label="val", color="red")
        plt.legend()
        ax = plt.gca()
        ax.set_ylim([-0.5, 1])
        plt.title("R^2")
        plt.savefig("plots/R2.png")
        plt.show()

        # Plot Mean Dist
        plt.plot(self.train_mean_dist, label="train", color="blue")
        plt.plot(self.val_mean_dist, label="val", color="red")
        plt.legend()
        plt.title("Mean Distance Error")
        plt.savefig("plots/MDE.png")
        plt.show()

    def print_step(self, predictions, actuals, dataset_type):
        mae, rmse, r2, mean_dist = self.compute_metrics(predictions, actuals, dataset_type)
        print(f"{dataset_type} \t MAE: {mae:>8f} \t RMSE: {rmse:>8f} \t R^2: {r2:>8f} \t Mean Dist: {mean_dist:>8f}")
        if dataset_type == "train":
            self.train_mae.append(mae)
            self.train_rmse.append(rmse)
            self.train_r2.append(r2)
            self.train_mean_dist.append(mean_dist)
        if dataset_type == "val":
            self.val_mae.append(mae)
            self.val_rmse.append(rmse)
            self.val_r2.append(r2)
            self.val_mean_dist.append(mean_dist)
            self.plot_metrics()
