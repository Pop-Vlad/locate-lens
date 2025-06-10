import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from utils import ModelType, CustomImageDataset, get_model, HaversineLoss, ModelEvaluator

test_dir = "./data/test"
model_type = ModelType.ViT
batch_size = 40  # CoAtNet: 4 # ViT: 8  # CNN: 40
display_step = 1000 // batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = CustomImageDataset(test_dir, "test", split_ratio=1.0, use_county=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
evaluator = ModelEvaluator(dataset, dataset)


def geo_dist(predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    result = np.zeros((predictions.shape[0], 1))
    for i in range(len(predictions)):
        lat1 = (predictions[i][0] - 0.5) * math.pi
        lon1 = (predictions[i][1] - 0.5) * 2 * math.pi
        lat2 = (actuals[i][0] - 0.5) * math.pi
        lon2 = (actuals[i][1] - 0.5) * 2 * math.pi
        dist = math.acos(
            math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * utils.R
        result[i] = dist
    return result


def add_results_to_county_dict(county_dict, img_names, dist, counties):
    for i in range(len(img_names)):
        county = counties[i]
        if county not in county_dict:
            county_dict[county] = []
        county_dict[county].append((img_names[i], dist[i][0]))


def get_top_by_county(county_dict, top_n):
    top = {}
    for county in county_dict:
        county_dict[county].sort(key=lambda x: x[1])
        top[county] = county_dict[county][:top_n]
    return {k: v for k, v in sorted(top.items())}


loss_func = utils.HaversineLoss()


def test(model):
    with torch.no_grad():
        i = 0
        county_dict = {}
        test_step_predictons = np.zeros((0, 2))
        test_step_actuals = np.zeros((0, 2))
        for _, data in tqdm(enumerate(dataloader)):
            # get the inputs and expected outputs; data is a list of [inputs, labels]
            inputs, actuals, img_names, counties = data
            inputs = inputs.to(device)
            actuals = actuals.to(device)

            # run model
            outputs = model(inputs)
            test_step_actuals = np.append(test_step_actuals, outputs.to("cpu").detach().numpy(), axis=0)
            test_step_predictons = np.append(test_step_predictons, actuals.to("cpu").detach().numpy(), axis=0)
            dist = geo_dist(outputs, actuals)
            add_results_to_county_dict(county_dict, img_names, dist, counties)

            i += 1

            if i % display_step == 0:
                print(f"Step {i}")
                mae, rmse, r2, mean_dist = evaluator.compute_metrics(test_step_predictons, test_step_actuals, "val")
                print(
                    f"test \t MAE: {mae:>8f} \t RMSE: {rmse:>8f} \t R^2: {r2:>8f} \t Mean Dist: {mean_dist:>8f}")
                print("Top 5 by county:")
                top = get_top_by_county(county_dict, 5)
                for county in top:
                    print(f"{county}: {top[county]}")


model = get_model(model_type)
model = model.to(device)
criterion = HaversineLoss()

if __name__ == '__main__':
    test(model)
