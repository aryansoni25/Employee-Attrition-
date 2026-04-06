from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from src.preprocessing import preprocess_pipeline



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


os.makedirs(MODEL_DIR, exist_ok=True)


def train_naive_bayes():
    
    X_train, X_test, y_train, y_test = preprocess_pipeline()

    
    model = GaussianNB()

    
    model.fit(X_train, y_train)

    
    preds = model.predict(X_test)

    
    print("\n🔹 Naive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))


    model_path = os.path.join(MODEL_DIR, "nb_model.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Model saved at: {model_path}")

    return model


if __name__ == "__main__":
    train_naive_bayes()