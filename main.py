import os
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from training_utils.dataset import CXRNoduleDataset, get_transform
from training_utils.train import train_one_epoch, collate_fn
import training_utils.utils as utils
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
data_path = './tbx11k/'
train_data = pd.read_csv(os.path.join(data_path, 'data_unhook.csv'))
train_data.drop(['width', 'height', 'target'], axis=1, inplace=True)
train_data = train_data.reindex(columns=['img_name', 'source', 'x', 'y', 'x2', 'y2'])
train_data = train_data.apply(pd.to_numeric, errors='coerce')
train_data = train_data.dropna()

# Define custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, boxes = self.df.iloc[idx]['img_name'], self.df.iloc[idx][1:].values.astype(float)
        img_path = os.path.join(self.img_dir, img_name)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = np.expand_dims(img, axis=-1).astype(np.float32)
        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        target = {'boxes': torch.tensor(boxes), 'labels': labels}
        if self.transform:
            img = self.transform(img)
        return img, target

# Define data transformations
data_transform = T.Compose([T.ToTensor()])

# Create dataset and dataloaders
train_dataset = CustomDataset(train_data, os.path.join(data_path, 'train'), transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Define and initialize the model
def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    return model

# Training function
def train_model(model, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device=device, epoch=epoch, print_freq=10)
    print("Training complete!")

# Prepare model, optimizer, and train the model
num_classes = 2  # Number of classes (nodule + background)
model = create_model(num_classes)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
train_model(model, optimizer)

# Save the trained model
torch.save(model.state_dict(), "model_v5_20")

# Inference and visualization (example code for prediction and visualization)

# Load an example image
image_path = './tbx11k/images/tb0036.png'
image = Image.open(image_path).convert('L')  # Convert to grayscale
image_tensor = data_transform(np.array(image)).unsqueeze(0).to(device)

# Put the model in evaluation mode and perform inference
model.eval()
with torch.no_grad():
    predictions = model(image_tensor)

# Filter predictions based on confidence score
filtered_predictions = [{'boxes': box, 'labels': label, 'scores': score}
                        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])
                        if score > 0.51]

# Visualize the image with bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image, cmap='gray')
for box in filtered_predictions:
    box_coords = (box['boxes'][0], box['boxes'][1]), box['boxes'][2] - box['boxes'][0], box['boxes'][3] - box['boxes'][1]
    rect = patches.Rectangle(*box_coords, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(box['boxes'][0], box['boxes'][1], f"Class: {box['labels']}, Score: {box['scores']:.2f}", color='r', backgroundcolor='white', fontsize=8)
plt.show()
