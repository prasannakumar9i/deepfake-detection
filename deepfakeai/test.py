import logging
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt  # <-- Add matplotlib import3
import seaborn as sns
import numpy as np

from deepfake.deepfakeai.utils import DeepfakeDataset
from deepfake.deepfakeai.models.base_model import CNN_ViT_LSTM

logger = logging.getLogger(__name__)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # Standardize input size
    transforms.RandomHorizontalFlip(p=0.5),        # Helps generalization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def test_model(config: dict[str, dict]):
    """
    Main function to test the model
    """
    image_dataset = config["imgdir"] / "test"
    models_dir = config["modelsdir"]
    result_dir = str(config["user_data_dir"] / "results")

    os.makedirs(result_dir, exist_ok=True)

    model_name = f"{config['model_name']}.pth"
    model_path = os.path.join(str(models_dir), model_name)

    # Load test dataset
    test_dataset = DeepfakeDataset(
        root_dir=str(image_dataset), 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = CNN_ViT_LSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate model
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.unsqueeze(1)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Prob for class "Fake"

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], output_dict=True)
    logger.info(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # 2. Classification Report Bar Chart
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['Real', 'Fake']
    scores = [[report[cls][metric] for metric in metrics] for cls in classes]
    scores = np.array(scores)
    x = np.arange(len(classes))
    width = 0.2

    plt.figure(figsize=(8,6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, scores[:, i], width, label=metric)

    plt.xticks(x + width, classes)
    plt.ylim(0, 1)
    plt.title("Classification Report Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "classification_report_metrics.png"))
    plt.close()

    # 3. Class Distribution Bar Plot
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_true, palette="pastel", label="True")
    sns.countplot(x=y_pred, palette="dark", alpha=0.5, label="Pred", hatch='//')
    plt.xticks(ticks=[0, 1], labels=["Real", "Fake"])
    plt.title("Class Distribution: True vs Predicted")
    plt.legend(["True", "Predicted"])
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "class_distribution.png"))
    plt.close()

    # 4. ROC Curve
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "roc_curve.png"))
    plt.close()
