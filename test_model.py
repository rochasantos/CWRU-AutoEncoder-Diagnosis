import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_model(model, dataloader, class_names, device="cuda"):
    # Lists to store actual and predicted labels
    y_true = []
    y_pred = []

    model.eval()  # Set model to evaluation mode

    # Disable gradient updates (faster evaluation)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    
    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\nConfusion Matrix:\n")
    print(cm_df)
