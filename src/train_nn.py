import torch
import torch.nn as nn
import torch.optim as optim

from src.preprocessing import preprocess_pipeline
from src.model import AttritionNN


def train_neural_network():
    X_train, X_test, y_train, y_test = preprocess_pipeline()

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)

    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Model
    model = AttritionNN(X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(50):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation
    with torch.no_grad():
        preds = model(X_test)
        preds = (preds > 0.5).float()

    accuracy = (preds.eq(y_test).sum() / len(y_test)).item()
    print("Neural Network Accuracy:", accuracy)

    # Save model
    torch.save(model.state_dict(), "models/nn_model.pth")

    return model


if __name__ == "__main__":
    train_neural_network()