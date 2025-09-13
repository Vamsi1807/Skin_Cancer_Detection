import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, device, 
                epochs=30, lr=3e-4, weight_decay=1e-4, patience=7, model_name="model"):
    
    criterion = nn.CrossEntropyLoss()  # ✅ no class weights
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True
    )

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.2e}")

        # ---- Training ----
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="weighted")
        print(f"  Train Loss: {np.mean(train_losses):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # ---- Validation ----
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")
        print(f"  Val Loss: {np.mean(val_losses):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Scheduler
        scheduler.step(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_save_name = f"best_{model_name}.pth"
            torch.save(model.state_dict(), model_save_name)
            print(f"  ✅ Saved best model: {model_save_name}")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break
