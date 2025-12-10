import os
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import random
import math
import copy

# ------------------------- SETTINGS -------------------------
BASE_DIR = r'D:\Diabetic-retinopathy-detection-main\dataset'
CLASS_FOLDER_NAMES = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
NUM_CLASSES = len(CLASS_FOLDER_NAMES)
CLASS_TO_INT = {class_name: i for i, class_name in enumerate(CLASS_FOLDER_NAMES)}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 0

INITIAL_TRAIN_EPOCHS = 5
FINETUNE_LAYER4_EPOCHS = 15
ADVANCED_FINETUNE_EPOCHS = 15
INITIAL_LR = 0.001

# Model save paths
INITIAL_MODEL_SAVE_PATH = 'diabetic_retinopathy_resnet50_initial_epochs5.pth'
FINETUNED_LAYER4_MODEL_SAVE_PATH = 'diabetic_retinopathy_resnet50_finetuned_epochs20.pth'
ADVANCED_MODEL_SAVE_PATH = 'diabetic_retinopathy_resnet50_advanced_best.pth'

# Early stopping
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001

# ------------------------- DATASET -------------------------
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return None, None
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        if not isinstance(label, torch.Tensor): label = torch.tensor(label, dtype=torch.long)
        return image, label

# ------------------------- GSA HYPERPARAMETER OPTIMIZATION -------------------------
def evaluate_hyperparameters(lr, weight_decay, model, train_loader, val_loader, device):
    temp_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    temp_model.to(device)
    temp_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = temp_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Validation loss
    temp_model.eval()
    val_loss_total, total_samples = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = temp_model(images)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item() * images.size(0)
            total_samples += labels.size(0)
    return val_loss_total / total_samples

def run_gsa(model, train_loader, val_loader, device):
    print("Starting GSA hyperparameter tuning for layer4 fine-tuning (lr, weight_decay)...")
    NUM_AGENTS = 5
    MAX_ITER = 5
    G0 = 100
    EPS = 1e-6
    LR_RANGE = (1e-5, 1e-3)
    WD_RANGE = (0.0, 0.01)
    agents = [[random.uniform(*LR_RANGE), random.uniform(*WD_RANGE)] for _ in range(NUM_AGENTS)]
    agents_fitness = [float('inf')]*NUM_AGENTS
    best_agent = None
    best_fitness = float('inf')

    for iteration in range(MAX_ITER):
        print(f"\nGSA Iteration {iteration+1}/{MAX_ITER}")
        for i, agent in enumerate(agents):
            lr, wd = agent
            fitness = evaluate_hyperparameters(lr, wd, model, train_loader, val_loader, device)
            agents_fitness[i] = fitness
            print(f"Agent {i+1}: lr={lr:.6f}, wd={wd:.6f}, val_loss={fitness:.4f}")
            if fitness < best_fitness:
                best_fitness = fitness
                best_agent = agent.copy()
        # Update agents (simplified)
        worst = max(agents_fitness)
        G = G0 * math.exp(-20*iteration/MAX_ITER)
        new_agents = []
        for i in range(NUM_AGENTS):
            force = [0.0,0.0]
            for j in range(NUM_AGENTS):
                if i==j: continue
                rij = math.sqrt(sum([(agents[j][d]-agents[i][d])**2 for d in range(2)])) + EPS
                f = G*(agents_fitness[j]-agents_fitness[i])/rij
                for d in range(2):
                    force[d] += random.random()*f*(agents[j][d]-agents[i][d])
            new_agent = [agents[i][d]+force[d] for d in range(2)]
            new_agent[0] = min(max(new_agent[0], LR_RANGE[0]), LR_RANGE[1])
            new_agent[1] = min(max(new_agent[1], WD_RANGE[0]), WD_RANGE[1])
            new_agents.append(new_agent)
        agents = new_agents
    print(f"\nGSA Completed - Best lr={best_agent[0]:.6f}, weight_decay={best_agent[1]:.6f}, val_loss={best_fitness:.4f}")
    return best_agent

# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gather image paths and labels
    all_image_paths, all_image_labels = [], []
    for class_name in CLASS_FOLDER_NAMES:
        class_path = os.path.join(BASE_DIR, class_name)
        if os.path.isdir(class_path):
            label = CLASS_TO_INT[class_name]
            current_class_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith('.png')]
            all_image_paths.extend(current_class_paths)
            all_image_labels.extend([label]*len(current_class_paths))
    all_image_paths_np = np.array(all_image_paths)
    all_image_labels_np = np.array(all_image_labels)

    # Split data
    train_val_paths, test_paths, train_val_labels_np, test_labels_np = train_test_split(all_image_paths_np, all_image_labels_np, test_size=0.15, stratify=all_image_labels_np, random_state=42)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_val_paths, train_val_labels_np, test_size=0.1765, stratify=train_val_labels_np, random_state=42)

    # Class weights
    class_counts_train = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_counts_train = np.where(class_counts_train==0,1,class_counts_train)
    class_weights = 1./class_counts_train
    class_weights = class_weights/np.sum(class_weights)*NUM_CLASSES
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Transforms
    train_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                           transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)])
    val_test_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)])

    # Datasets & loaders
    train_dataset = DiabeticRetinopathyDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = DiabeticRetinopathyDataset(val_paths, val_labels, transform=val_test_transforms)
    test_dataset = DiabeticRetinopathyDataset(test_paths, test_labels_np, transform=val_test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Load initial model if exists
    if os.path.exists(INITIAL_MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(INITIAL_MODEL_SAVE_PATH, map_location=device))
        print("Initial model loaded.")

    # Freeze all layers except layer4 and fc
    for name,param in model.named_parameters():
        if name.startswith('fc.') or name.startswith('layer4.'):
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to(device)

    # -------------------- GSA Hyperparameter Tuning --------------------
    best_lr, best_wd = run_gsa(model, train_loader, val_loader, device)

    # -------------------- Full Layer4 Fine-Tuning --------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=best_lr, weight_decay=best_wd)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(FINETUNE_LAYER4_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data,1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss/total
        epoch_acc = correct/total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_loss_total, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()*images.size(0)
                _, predicted_val = torch.max(outputs.data,1)
                correct_val += (predicted_val==labels).sum().item()
                total_val += labels.size(0)
        val_loss_epoch = val_loss_total/total_val
        val_acc_epoch = correct_val/total_val
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)
        scheduler.step(val_loss_epoch)
        print(f"Epoch {epoch+1}/{FINETUNE_LAYER4_EPOCHS} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} - Val Loss: {val_loss_epoch:.4f}, Acc: {val_acc_epoch:.4f}")
        if val_loss_epoch < best_val_loss-MIN_DELTA:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            torch.save(model.state_dict(), FINETUNED_LAYER4_MODEL_SAVE_PATH)
            print(f"Validation improved. Model saved to {FINETUNED_LAYER4_MODEL_SAVE_PATH}")
        else:
            patience_counter +=1
            if patience_counter>=EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(FINETUNED_LAYER4_MODEL_SAVE_PATH))
    model.eval()

    # -------------------- TEST EVALUATION --------------------
    all_labels, all_preds = [], []
    test_loss_total, correct_test, total_test = 0.0,0,0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_total += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data,1)
            correct_test += (predicted==labels).sum().item()
            total_test += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    print(f"Test Loss: {test_loss_total/total_test:.4f}, Accuracy: {correct_test/total_test:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[INT_TO_CLASS[i] for i in range(NUM_CLASSES)]))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[INT_TO_CLASS[i] for i in range(NUM_CLASSES)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
