import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the metrics 
df = pd.read_csv("logs/plain/train_metrics.csv")
sns.set(style="whitegrid")

# Train vs Validation Loss
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss")
sns.lineplot(data=df, x="epoch", y="val_loss", label="Validation Loss")
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.legend()
plt.tight_layout()
plt.savefig("plots/train_val_loss.png")

#Lambda over Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="epoch", y="lambda_val", label="Lambda Value", color="purple")
plt.title("Lambda Value over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Lambda")
plt.tight_layout()
plt.savefig("plots/lambda_over_epochs.png")

#Validation Loss vs Lambda
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="lambda_val", y="val_loss", label="val loss", color="blue")
plt.title("Validation Loss vs Lambda Value")
plt.xlabel("Lambda Value")
plt.ylabel("Validation Loss")
plt.tight_layout()
plt.savefig("plots/val_loss_vs_lambda.png")