import torch.nn.functional
from torchvision.transforms import transforms

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class Adapter:

    # Translation layer between the UI and the model

    def __init__(self, modelType):
        self.transform = transforms.Compose([transforms.ToTensor(), utils.image_transform])
        self.model = utils.get_model(modelType)
        self.model.to(device)

    def changeModel(self, modelType):
        self.model = utils.get_model(modelType)
        self.model.to(device)

    def runForImage(self, pil_image):
        # Run the model on a single image
        input = self.transform(pil_image)
        input = input.unsqueeze(0)
        input = input.to(device)
        with torch.no_grad():
            output = self.model(input)
            np_arr = output.cpu().detach().numpy()
            result = np_arr[0].astype(float)
        return self.postProcess(result)

    def postProcess(self, result):
        # Post process the output of the model
        result = utils.to_degrees(result)
        result[0] = (result[0] + 90) % 180 - 90
        result[1] = (result[1] + 180) % 360 - 180
        result[0] = min(max(result[0], -85), 85)
        result[1] = min(max(result[1], -180), 180)
        return result


