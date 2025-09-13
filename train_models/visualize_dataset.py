import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter

# --- Assume you already loaded your train_df, val_df, and sampler ---

# 1️⃣ Original dataset distribution
sns.countplot(x=train_df['label'])
plt.title("Original Class Distribution (Train)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 2️⃣ Distribution after WeightedRandomSampler (simulate one epoch)
balanced_samples = []
for idx in torch.utils.data.DataLoader(
    train_df['label'].values, 
    batch_size=32, 
    sampler=sampler
):
    balanced_samples.extend(idx.numpy())

balanced_counts = Counter(balanced_samples)

plt.bar(balanced_counts.keys(), balanced_counts.values())
plt.title("Balanced Distribution (One Epoch via Sampler)")
plt.xlabel("Class")
plt.ylabel("Sample Count")
plt.show()
