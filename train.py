import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import os

from model import SimpleNet

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# input transformations allow the model to randomly view a slightly warped version of the iniial data
# it is not creating new images it is just applying a math filter before feeding it in
# this way it might be cropped, sharpened, flipped etc
# since the model is not viewing the same repeated pixels it will be forced to generalize rules
# which is a cure for overfitting (memorizing pixcels instead of learning rules)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # random crop after padding — teaches translation invariance (position shifts)

    transforms.RandomHorizontalFlip(),
    # CIFAR-10 classes are left-right symmetric — free variety, safe for this dataset
    # (would NOT be safe for e.g. digit/text datasets where flipping changes meaning)

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    # randomly shifts brightness/contrast/saturation/hue within a small range —
    # teaches the model not to rely on exact lighting/color, which varies a lot
    # in real photos. Kept mild (small ranges) since aggressive color shifts
    # can distort object identity for color-dependent classes (e.g. certain birds/fruit)

    transforms.RandomRotation(10),
    # small rotation (±10°) — real-world photos are rarely perfectly level;
    # kept small since CIFAR-10 images are tiny (32x32) and large rotations
    # destroy too much spatial information at that resolution

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    # (Cutout-style) randomly blacks out a small rectangular patch after normalization —
    # forces the model to use multiple cues rather than over-relying on one salient
    # region (e.g. always keying off the same corner of the image). Kept to a small
    # scale + 25% probability so it's a mild regularizer, not overwhelming.
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_loaders():
    # Ensure train and test data directories exist
    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    # Load the full training set
    dataset_train_full = datasets.CIFAR10(
        root=config['paths']['train_dir'],
        train=True,
        download=True,
        transform=transform
    )
    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=transform
    )

    # Split train into train/val (e.g., 90% train, 10% val)
    val_split = config['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    dataset_train, dataset_val = random_split(dataset_train_full, [n_train, n_val])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    return dataloader_train, dataloader_val, dataloader_test


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# updated here as well!
def train(model, dataloader_train, dataloader_val, criterion, optimizer, device):
    model.train()
    epochs = config['hyperparameters']['epochs']
    best_val_acc = 0.0  # track the best validation accuracy seen so far

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(
            dataloader_train,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': avg_loss})
        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        print(f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")

        # save a checkpoint only when this epoch beats every previous one
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)
            torch.save(model.state_dict(), config['paths']['model_path'])
            print(f"  New best (val acc {val_acc:.2f}%) — checkpoint saved")

        model.train()  # evaluate() sets eval mode, switch back before next epoch

    print(f'Finished Training. Best val acc: {best_val_acc:.2f}%')

# Function to test the neural network and print accuracy
def test(model, dataloader_test, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total number of samples
    with torch.no_grad():  # No need to compute gradients during testing (save memory)
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Update total count
            correct += (predicted == labels).sum().item()  # Update correct count
    # Print accuracy as a percentage
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')


def main():
    # Select device: use GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders for training and testing
    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    # Create the neural network and move it to the selected device
    model = SimpleNet().to(device)

    # Define the loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (SGD with learning rate and momentum from config)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['hyperparameters']['lr'],
        momentum=config['hyperparameters']['momentum']
    )

    train(model, dataloader_train, dataloader_val, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    # Save the trained model's parameters to a file
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)  # Create weights dir if missing
    torch.save(model.state_dict(), config['paths']['model_path']) # Save the model's parameters to a pth file
    print(f"Model saved to {config['paths']['model_path']}")

if __name__ == '__main__':
    main()
