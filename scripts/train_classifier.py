import torch
import torch.nn as nn
import torch.optim as optim

def train_classifier(model, dataloader, lr=1e-3, epochs=100, saved_path="saved_models/vae_clas.pth", device="cuda"):
    # Optimizer configuration and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        train_loss = 0
        correct, total = 0, 0

        for batch in dataloader:
            input_img, labels = batch  
            input_img, labels = input_img.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_img) 
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), saved_path)