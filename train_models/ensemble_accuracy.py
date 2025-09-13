import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- Load models ---
from models.densenet121 import get_densenet121
from models.mobilenetv2 import get_mobilenet_v2
from models.xception import get_xception
from models.resnet50 import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = {
    "densenet121": "best_densenet.pth",
    "mobilenetv2": "best_mobilenetv2.pth",
    "xception": "best_xception.pth",
    "resnet50": "best_resnet50.pth"
}

label_map = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}

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

# --- Ensemble evaluation ---
def evaluate_ensemble(models, dataloader, device, tta=5):
    all_preds, all_targets = [], []

    for images, labels in tqdm(dataloader, desc="Evaluating Ensemble"):
        images, labels = images.to(device), labels.to(device)
        batch_preds = []

        for _ in range(tta):
            model_preds = []
            for model in models:
                with torch.no_grad():
                    out = model(images)
                    probs = F.softmax(out, dim=1)
                    model_preds.append(probs.cpu().numpy())

            ensemble_pred = np.mean(model_preds, axis=0)
            batch_preds.append(ensemble_pred)

        final_probs = np.mean(batch_preds, axis=0)  # [batch, num_classes]
        preds = np.argmax(final_probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    return acc, f1


# --- Run evaluation ---
if __name__ == "__main__":
    from train_models.dataset import get_dataloaders

    PROJECT_DIR = r"C:\Users\vamsi\Desktop\4-1 mini project\skin-cancer-detection"
    _, val_loader = get_dataloaders(PROJECT_DIR, batch_size=32)

    acc, f1 = evaluate_ensemble(models_list, val_loader, device, tta=5)
    print(f"\n📊 Ensemble Accuracy: {acc*100:.2f}%")
    print(f"📊 Ensemble F1-score: {f1*100:.2f}%")

