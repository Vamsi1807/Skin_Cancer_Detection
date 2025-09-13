import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import torch
from train_models.dataset import get_dataloaders
from models.resnet50 import get_resnet50
from train import train_model

if __name__ == "__main__":
    PROJECT_DIR = r"C:\Users\Vamsi\Desktop\4-1 mini project\skin-cancer-detection"
    train_loader, val_loader = get_dataloaders(PROJECT_DIR, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_resnet50(num_classes=7).to(device)

    train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4,model_name="resnet50")

