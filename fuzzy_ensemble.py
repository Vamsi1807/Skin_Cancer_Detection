import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import pandas as pd
import sys, os

# --- Import dataset + models ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# --- Load validation data ---
_, val_loader = get_dataloaders(PROJECT_DIR, batch_size=32)

# --- Load models ---
def load_model(model_fn, path):
    model = model_fn(num_classes=num_classes)
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

# --- Fuzzy Rank Aggregation Ensemble ---
def fuzzy_ensemble_predict(models, images):
    all_model_scores = []

    for model in models:
        with torch.no_grad():
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            # Convert probs -> ranks
            ranks = np.argsort(np.argsort(-probs, axis=1), axis=1) + 1  # rank 1 = best
            fuzzy_scores = 1.0 / ranks  # fuzzy membership score
            all_model_scores.append(fuzzy_scores)

    # Aggregate scores across models
    agg_scores = np.sum(all_model_scores, axis=0)
    preds = np.argmax(agg_scores, axis=1)
    return preds


if __name__ == "__main__":
    # --- Evaluate Fuzzy Ensemble ---
    all_preds, all_targets = [], []

    for images, labels in tqdm(val_loader, desc="Fuzzy Ensemble Evaluation"):
        images, labels = images.to(device), labels.to(device)
        preds = fuzzy_ensemble_predict(models_list, images)
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    # --- Metrics ---
    acc = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average="weighted") * 100
    report = classification_report(all_targets, all_preds, target_names=label_map.values())

    print(f"\n📊 Fuzzy Ensemble Accuracy: {acc:.2f}%")
    print(f"📊 Fuzzy Ensemble F1-score: {f1:.2f}%")
    print("\nDetailed Report:\n", report)

    # --- Save results ---
    df_report = pd.DataFrame(classification_report(all_targets, all_preds, target_names=label_map.values(), output_dict=True)).transpose()
    df_report.to_csv(os.path.join(PROJECT_DIR, "fuzzy_ensemble_metrics.csv"))
    print(f"\n✅ Saved detailed results to fuzzy_ensemble_metrics.csv")
