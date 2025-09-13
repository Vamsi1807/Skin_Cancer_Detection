import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saved model paths
model_paths = {
    "densenet121": "best_densenet.pth",
    "mobilenetv2": "best_mobilenetv2.pth",
    "xception": "best_xception.pth",
    "resnet50": "best_resnet50.pth"
}

# Class labels
label_map = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}

# --- Load models ---
from models.densenet121 import get_densenet121
from models.mobilenetv2 import get_mobilenet_v2
from models.xception import get_xception
from models.resnet50 import get_resnet50

def load_model(model_fn, path):
    model = model_fn(num_classes=len(label_map))
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

models_list = [
    load_model(get_densenet121, model_paths["densenet121"]),
    load_model(get_mobilenet_v2, model_paths["mobilenetv2"]),
    load_model(get_xception, model_paths["xception"]),
    load_model(get_resnet50, model_paths["resnet50"])
]

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- TTA + Ensemble ---
def predict_ensemble_tta(models, img_path, device, tta=5):
    img = Image.open(img_path).convert("RGB")
    preds_all = []

    for _ in range(tta):
        aug_img = transform(img)
        aug_img = aug_img.unsqueeze(0).to(device)

        model_preds = []
        for model in models:
            with torch.no_grad():
                out = model(aug_img)
                probs = F.softmax(out, dim=1).cpu().numpy()
                model_preds.append(probs)

        ensemble_pred = np.mean(model_preds, axis=0)
        preds_all.append(ensemble_pred)

    avg_pred = np.mean(preds_all, axis=0).squeeze()
    return avg_pred

# --- Run prediction ---
img_path = r"C:\Users\vamsi\Downloads\archive\dataverse_files\ISIC2018_Task3_Test_Images\ISIC_0036056.jpg"
probs = predict_ensemble_tta(models_list, img_path, device, tta=8)

# --- Get Top-3 classes ---
top3_idx = probs.argsort()[-3:][::-1]
print("\n🔮 Predictions (Top-3):")
for idx in top3_idx:
    print(f"  {label_map[idx]} : {probs[idx]*100:.2f}%")

# --- Final decision ---
best_class = label_map[top3_idx[0]]
print(f"\n✅ Final Predicted class: {best_class}")
