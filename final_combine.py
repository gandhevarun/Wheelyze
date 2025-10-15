#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# In[10]:


# Validation/test transform
val_transform = transforms.Compose([
    transforms.Resize((300, 300)),  # same as B3 training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# In[11]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# EfficientNet-B3 models
binary_model = models.efficientnet_b3(weights=None)
binary_model.classifier[1] = nn.Linear(binary_model.classifier[1].in_features, 2)

defective_model = models.efficientnet_b3(weights=None)
defective_model.classifier[1] = nn.Linear(defective_model.classifier[1].in_features, 5)

good_model = models.efficientnet_b3(weights=None)
good_model.classifier[1] = nn.Linear(good_model.classifier[1].in_features, 5)

# ResNet18 model
resnet = models.resnet18(weights=None)
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet.fc.in_features, 10)
)

# Load trained weights
binary_model.load_state_dict(torch.load("binary_best.pth", map_location=device))
defective_model.load_state_dict(torch.load("defectiveFinal.pth", map_location=device))
good_model.load_state_dict(torch.load("good_best.pth", map_location=device))
resnet.load_state_dict(torch.load("tyre_resnet18_best.pth", map_location=device))

# Move to device and eval mode
binary_model.to(device).eval()
defective_model.to(device).eval()
good_model.to(device).eval()
resnet.to(device).eval()

print("All models loaded successfully")


# In[28]:


def predict_tyre_custom_ensemble(image_path):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Stage 1: EfficientNet binary
        out_bin = binary_model(image)
        _, pred_bin = torch.max(out_bin, 1)  # 0=Defective, 1=Good

        if pred_bin.item() == 0:  # Defective (1–5)
            eff_probs = F.softmax(defective_model(image), dim=1)
            eff_pred = eff_probs.argmax(dim=1).item() + 1
            valid_range = range(1, 6)
        else:  # Good (6–10)
            eff_probs = F.softmax(good_model(image), dim=1)
            eff_pred = eff_probs.argmax(dim=1).item() + 6
            valid_range = range(6, 11)

        # ResNet full prediction
        res_out = resnet(image)
        _, res_pred = torch.max(res_out, 1)
        res_pred = res_pred.item() + 1

        # Check if ResNet prediction is in EfficientNet's range
        if res_pred in valid_range:
            # Proper average
            avg = (eff_pred + res_pred) / 2
            # Round: >=0.5 → up, <0.5 → down
            if (avg - int(avg)) >= 0.5:
                ensemble_pred = int(avg) + 1
            else:
                ensemble_pred = int(avg)
        else:
            # ResNet disagrees → use EfficientNet only
            ensemble_pred = f"{eff_pred}*"

    return {
        "EfficientNet_prediction": eff_pred,
        "ResNet_prediction": res_pred,
        "Ensemble_prediction": ensemble_pred
    }


# In[ ]:


if __name__ == "__main__":
    test_image = r"tyresample2.jpg"
    predictions = predict_tyre_custom_ensemble(test_image)
    print(f"Test Image: {test_image}")
    print(f"EfficientNet Prediction: {predictions['EfficientNet_prediction']}")
    print(f"ResNet Prediction: {predictions['ResNet_prediction']}")
    print(f"Ensemble Prediction: {predictions['Ensemble_prediction']}")