# training_utils.py
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, criterion):
    """
    Trains the model for one epoch and returns the average loss.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    """
    Evaluates the model on the given dataloader and returns the accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def self_training_step(model, dataloader, optimizer, criterion, num_epochs=50):
    """
    Applies a self-training step on the given dataloader.
    For each batch, it computes pseudo-labels using argmax on the logits
    and updates the model by minimizing the cross-entropy between the outputs and these pseudo-labels.
    It prints the average loss and accuracy on the intermediate domain at each epoch.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for inputs, _ in dataloader:  # ignoring real labels
            outputs = model(inputs)
            pseudo_labels = outputs.argmax(dim=1)
            loss = criterion(outputs, pseudo_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
        avg_loss = epoch_loss / count
        acc = evaluate(model, dataloader)
        print(f"  Self-training epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")
    return model