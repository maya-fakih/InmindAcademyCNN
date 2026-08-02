import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import json

import os

from model import ResNet18

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# input transformations allow the model to randomly view a slightly warped version of the iniial data
# it is not creating new images it is just applying a math filter before feeding it in
# this way it might be cropped, sharpened, flipped etc
# since the model is not viewing the same repeated pixels it will be forced to generalize rules
# which is a cure for overfitting (memorizing pixcels instead of learning rules)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),\
    # we added padding then cropped out the same size so its fine we dont have different size input which is an issue for resnet
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
        transform=transform_train
    )
    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=transform_test
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

def save_checkpoint(model, optimizer, epoch, best_val_acc, path, history=None):
    """Saves everything needed to resume training exactly where it left off:
    model weights, optimizer state (momentum buffers), current epoch number,
    best val accuracy seen so far, and per-epoch history for later plotting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
    }, path)

# IMPORTANT FIX
# tracking the best validation accuracy and saving a checkpoint to not loose a better model among many epochs
def train(model, dataloader_train, dataloader_val, criterion, optimizer, device,
          start_epoch=0, best_val_acc=0.0, history=None):
    # tracks to save history for save checkpoint :D
    total_epochs = config['hyperparameters']['epochs']
    latest_path = 'weights/latest.pth'
    best_path = 'weights/best.pth'
    if history is None:
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(
            dataloader_train,
            desc=f"Epoch {epoch+1}/{total_epochs}",
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
        print(f"Epoch {epoch+1}/{total_epochs} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        save_checkpoint(model, optimizer, epoch, best_val_acc, latest_path, history)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, best_path, history)
            print(f"  New best (val acc {val_acc:.2f}%) — best.pth updated")

    print(f'Finished Training. Best val acc: {best_val_acc:.2f}%')
    return best_val_acc, history

def evaluate(model, dataloader, criterion, device, collect_predictions=False):
    """Runs one pass over dataloader, returns avg loss + accuracy.
    If collect_predictions=True, also returns the raw predictions/labels
    lists, so a second pass isn't needed just to build a confusion matrix."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if collect_predictions:
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    if collect_predictions:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy

def compute_classification_metrics(all_preds, all_labels, class_names):
    """Takes predictions already collected by evaluate(), returns per-class
    precision/recall/F1 (dict) and confusion matrix (2D list)."""
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm.tolist()

def save_metrics_json(test_loss, test_acc, best_val_acc, history, report, cm, class_names, path='logs/metrics.json'):
    """Writes one JSON file containing everything scripts/generate_report.py
    needs to build plots and a markdown summary — keeps training and
    reporting as two separate, independent steps."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'history': history,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names,
        }, f, indent=2)
    print(f"Saved metrics to {path}")

def main():
    # Select device: use GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders for training and testing
    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    # Create the neural network and move it to the selected device
    model = ResNet18(num_classes=10).to(device)

    # Define the loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (SGD with learning rate and momentum from config)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['hyperparameters']['lr'],
        momentum=config['hyperparameters']['momentum']
    )

    # --- Resume logic: if a checkpoint exists at the configured resume path, load it ---
    start_epoch = 0
    best_val_acc = 0.0
    history = None
    resume_path = config['paths'].get('resume_from')

    if resume_path and os.path.exists(resume_path):
        print(f"Found checkpoint at {resume_path}, resuming...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("No optimizer state in checkpoint (converted from an old run) — optimizer starts fresh")
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint.get('history')
        print(f"Resumed from epoch {start_epoch}, best val acc so far: {best_val_acc:.2f}%")
    else:
        print("No checkpoint to resume from, starting fresh")

    best_val_acc, history = train(model, dataloader_train, dataloader_val, criterion, optimizer, device,
                                   start_epoch=start_epoch, best_val_acc=best_val_acc, history=history)

    # --- Load best weights before final test evaluation, not whatever's left in memory ---
    best_checkpoint = torch.load('weights/best.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, dataloader_test, criterion, device, collect_predictions=True)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    report, cm = compute_classification_metrics(test_preds, test_labels, class_names)
    save_metrics_json(test_loss, test_acc, best_val_acc, history, report, cm, class_names)

if __name__ == '__main__':
    main()
