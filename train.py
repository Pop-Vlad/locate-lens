import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *

train_dir = "./data/train"
model_type = ModelType.ViT
batch_size = 8  # CoAtNet: 4 # ViT: 8  # CNN: 8
num_epochs = 1
learning_rate = 1e-6  # CoAtNet: 1e-6 to 1e-9 # ViT: 1e-5 to 1e-7 # CNN: 1e-5 to 1e-7
display_step = 2000 // batch_size
save_step = 2000 // batch_size
run_validation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = CustomImageDataset(train_dir, "train", split_ratio=0.9 if run_validation else 1.0)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomImageDataset(train_dir, "val", split_ratio=0.9 if run_validation else 1.0)
val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = get_model(model_type)
model = model.to(device)

# Train using MSELoss initially, then switch to HaversineLoss
# criterion = nn.MSELoss()
criterion = HaversineLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

evaluator = ModelEvaluator(train_dataset, val_dataset)


def validate(model):
    val_step_predictons = np.zeros((0, 2))
    val_step_actuals = np.zeros((0, 2))
    for j, data in tqdm(enumerate(val_dataloader)):
        # get the inputs and expected outputs; data is a list of [inputs, labels]
        inputs, actuals = data
        inputs = inputs.to(device)
        actuals = actuals.to(device)
        with torch.no_grad():
            # run model
            outputs = model(inputs)
            val_step_actuals = np.append(val_step_actuals, outputs.to("cpu").detach().numpy(), axis=0)
            val_step_predictons = np.append(val_step_predictons, actuals.to("cpu").detach().numpy(), axis=0)
        if j >= display_step - 1:
            break
    evaluator.print_step(val_step_predictons, val_step_actuals, "val")


def train(model, loss_function, optimizer, num_epochs=10):
    i = 0
    step_predictons = np.zeros((0, 2))
    step_actuals = np.zeros((0, 2))
    mean_running_loss = 0.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for _, data in tqdm(enumerate(train_dataloader)):
            # get the inputs and expected outputs; data is a list of [inputs, labels]
            inputs, actuals = data
            inputs = inputs.to(device)
            actuals = actuals.to(device)

            optimizer.zero_grad()
            # run model
            outputs = model(inputs)
            # compute loss
            loss = loss_function(outputs, actuals)
            loss.backward()
            optimizer.step()
            mean_running_loss += loss.item()
            i += 1

            step_actuals = np.append(step_actuals, outputs.to("cpu").detach().numpy(), axis=0)
            step_predictons = np.append(step_predictons, actuals.to("cpu").detach().numpy(), axis=0)

            if i % display_step == 0:
                mean_running_loss = mean_running_loss / display_step
                print(f"\nEpoch {epoch}: Step {i}: Loss: {mean_running_loss}")
                mean_running_loss = 0.0
                evaluator.print_step(step_predictons, step_actuals, "train")

                # Clear the lists to track the next step
                step_predictons = np.zeros((0, 2))
                step_actuals = np.zeros((0, 2))

                # Evaluate on validation set
                if run_validation:
                    validate(model)

            if i % save_step == 0:
                torch.save(model.state_dict(), "./trained_models/" + str(model_type.name) + ".pth")
                print("Saved model state to " + str(model_type.name) + ".pth")

    print('Finished Training')


if __name__ == '__main__':
    train(model, criterion, optimizer, num_epochs=num_epochs)
