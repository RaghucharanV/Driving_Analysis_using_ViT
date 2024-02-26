import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
import pandas as pd

#Seeding
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seeds()
#Verifing Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
#Pre-trained
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

#class_labels
class_names = [
    'safe driving', 'texting - right', 'talking on the phone - right',
    'texting - left', 'talking on the phone - left', 'operating the radio',
    'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger'
]  
#Fine-tuning
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)


#Training Data


#Data-Loader
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir:str, 
    test_dir:str, 
    transform: transforms.Compose, 
    batch_size: int, 
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

#Transformers Model
pretrained_vit_transforms = pretrained_vit_weights.transforms()


#spliting Data
import os
import random
from shutil import copyfile

root_dir = 'test_dataset'

train_dir = 'trains'
test_dir = 'tests'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        num_images = len(images)

        random.shuffle(images)

        num_train_images = int(num_images * 0.8)
        train_images = images[:num_train_images]
        test_images = images[num_train_images:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(train_dir, class_folder, img)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            copyfile(src_path, dst_path)

        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(test_dir, class_folder, img)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            copyfile(src_path, dst_path)

#Dir
train_dir = 'trains'
test_dir = 'tests'

#Loading Data
train_dataloader_pretrained,test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,test_dir=test_dir,transform=pretrained_vit_transforms,batch_size=32)

#Optimizer and Loss
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

#Training Data
set_seeds()
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


#Model Training
pretrained_vit_results = train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=9,
                                      device=device)

model = pretrained_vit

#model classify
def classify_image(image):
    # Create transformation for image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Predict
    model.to(device)
    model.eval()

    with torch.inference_mode():
        transformed_img = image_transform(image).unsqueeze(dim=0)
        target_image_pred = model(transformed_img.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    return class_names[target_image_pred_label]

st.title("Driver State Classification App")


# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Classify the image
    result = classify_image(image)
    
    st.success(f"Prediction: {result}")

# Display sample images for users to choose from
st.sidebar.title("Choose a Sample Image")
sample_images =['img_102025.jpg','img_102119.jpg','']
selected_sample = st.sidebar.selectbox("Select a sample image", sample_images)

# Load and display the selected sample image
sample_image_path = selected_sample
sample_image = Image.open(sample_image_path)
st.sidebar.image(sample_image, caption="Sample Image.", use_column_width=True)

# Classify the sample image
sample_result = classify_image(sample_image)
st.sidebar.success(f"Prediction for Sample Image: {sample_result}")
