import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import numpy as np

from train_models.dataset import get_dataloaders
from models.densenet121 import get_densenet121
from models.mobilenetv2 import get_mobilenet_v2
from models.xception import get_xception
from models.resnet50 import get_resnet50

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = r"C:\Users\vamsi\Desktop\4-1 mini project\skin-cancer-detection"

model_paths = {
    "densenet121": "best_densenet.pth",
    "mobilenetv2": "best_mobilenetv2.pth",
    "xception": "best_xception.pth",
    "resnet50": "best_resnet50.pth"
}

label_map = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
num_classes = len(label_map)

# --- Function to load model ---
def load_model(model_fn, path):
    model = model_fn(num_classes=num_classes)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# --- Evaluate single model ---
def evaluate_model(model, dataloader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average="weighted") * 100
    report = classification_report(
        all_targets, all_preds, target_names=label_map.values(), output_dict=True
    )
    return acc, f1, report

# --- Main ---
if __name__ == "__main__":
    # ✅ Force num_workers=0 for evaluation (avoids Windows spawn error)
    _, val_loader = get_dataloaders(PROJECT_DIR, batch_size=32, seed=42)
    val_loader.num_workers = 0  

    results = []
    for name, path in model_paths.items():
        print(f"\n🔍 Evaluating {name}...")
        if name == "densenet121":
            model = load_model(get_densenet121, path)
        elif name == "mobilenetv2":
            model = load_model(get_mobilenet_v2, path)
        elif name == "xception":
            model = load_model(get_xception, path)
        elif name == "resnet50":
            model = load_model(get_resnet50, path)

        acc, f1, report = evaluate_model(model, val_loader, device)

        print(f"📊 {name} Accuracy: {acc:.2f}% | F1-score: {f1:.2f}%")
        df_report = pd.DataFrame(report).transpose()
        print(df_report)

        results.append({"Model": name, "Accuracy": acc, "F1_score": f1})

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(PROJECT_DIR, "model_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
