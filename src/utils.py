import numpy as np
import joblib
import torch

from src.model import AttritionNN
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "nb_model.pkl")


def load_nb_model():
    return joblib.load("models/nb_model.pkl")


def load_scaler():
    return joblib.load("models/scaler.pkl")


def load_nn_model(input_size):
    model = AttritionNN(input_size)
    model.load_state_dict(torch.load("models/nn_model.pth"))
    model.eval()
    return model


def predict_nb(data):
    model = load_nb_model()
    scaler = load_scaler()

    expected_features = scaler.n_features_in_

    if len(data) != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {len(data)}")

    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    return int(model.predict(data)[0])


def predict_nn(data, input_size):
    model = load_nn_model(input_size)
    scaler = load_scaler()

    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    data = torch.FloatTensor(data)

    with torch.no_grad():
        pred = model(data)
        return int((pred > 0.5).float().item())