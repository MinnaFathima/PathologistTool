import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from transformers import ViTModel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# ---------------------------
# 1. Load CSV and dataset
# ---------------------------
data_dir = "./train"               # folder where .tif images are extracted
labels_csv = "./train_labels.csv"  # labels file

df = pd.read_csv(labels_csv)
df["id"] = df["id"].astype(str) + ".tif"

print("✅ Number of images:", len(df))
print(df.head())


# ---------------------------
# 2. Transforms for CNN & ViT
# ---------------------------
cnn_transform = transforms.Compose([
    transforms.Resize((96, 96)),   # smaller for CPU
    transforms.ToTensor(),
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------
# 3. Custom Dataset
# ---------------------------
class DualTransformDataset(Dataset):
    def __init__(self, df, img_dir, cnn_transform, vit_transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cnn_transform = cnn_transform
        self.vit_transform = vit_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id"])
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        img = Image.open(img_path).convert("RGB")
        cnn_img = self.cnn_transform(img)
        vit_img = self.vit_transform(img)

        return cnn_img, vit_img, label


# ---------------------------
# 4. Train/Val Split
# ---------------------------
train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_df, val_df = random_split(df, [train_size, val_size])

train_dataset = DualTransformDataset(train_df.dataset.iloc[train_df.indices], data_dir, cnn_transform, vit_transform)
val_dataset   = DualTransformDataset(val_df.dataset.iloc[val_df.indices], data_dir, cnn_transform, vit_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


# ---------------------------
# 5. Hybrid CNN + ViT Model
# ---------------------------
class HybridCNNViT(nn.Module):
    def __init__(self):
        super(HybridCNNViT, self).__init__()
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes
        )

    def forward(self, cnn_x, vit_x):
        cnn_feat = self.cnn(cnn_x)                 # (batch, 512)
        vit_feat = self.vit(vit_x).pooler_output   # (batch, 768)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.fc(combined)


# ---------------------------
# 6. Training Setup
# ---------------------------
device = torch.device("cpu")  # Force CPU
model = HybridCNNViT().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ---------------------------
# 7. Training Loop
# ---------------------------
def train_model(model, train_loader, val_loader, epochs=1):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for cnn_imgs, vit_imgs, labels in tqdm(train_loader, desc="Training"):
            cnn_imgs, vit_imgs, labels = cnn_imgs.to(device), vit_imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(cnn_imgs, vit_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for cnn_imgs, vit_imgs, labels in val_loader:
                cnn_imgs, vit_imgs, labels = cnn_imgs.to(device), vit_imgs.to(device), labels.to(device)

                outputs = model(cnn_imgs, vit_imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    return history


history = train_model(model, train_loader, val_loader, epochs=1)  # keep epochs=1 for CPU test


# ---------------------------
# 8. Evaluation (Confusion Matrix + ROC)
# ---------------------------
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for cnn_imgs, vit_imgs, labels in val_loader:
        cnn_imgs, vit_imgs, labels = cnn_imgs.to(device), vit_imgs.to(device), labels.to(device)
        outputs = model(cnn_imgs, vit_imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Tumor"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Validation Set")
plt.show()

fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# ---------------------------
# 9. Save Model
# ---------------------------
torch.save(model.state_dict(), "hybrid_cnn_vit.pth")
print("✅ Model weights saved as hybrid_cnn_vit.pth")
