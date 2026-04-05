from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from src.preprocessing import preprocess_pipeline


# 🔹 Base path (IMPORTANT for Docker + API)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def train_naive_bayes():
    # Load processed data
    X_train, X_test, y_train, y_test = preprocess_pipeline()

    # Initialize model
    model = GaussianNB()

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluation
    print("\n🔹 Naive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    # Save model (FIXED PATH)
    model_path = os.path.join(MODEL_DIR, "nb_model.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Model saved at: {model_path}")

    return model


if __name__ == "__main__":
    train_naive_bayes()