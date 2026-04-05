# 💼 Employee Attrition Prediction System

A Machine Learning-based web application that predicts whether an employee is likely to leave the organization based on various factors such as job role, income, experience, and work conditions.

---

## 🚀 Features

- Predict employee attrition using ML models
- Supports **Naive Bayes** and **Neural Network (PyTorch)**
- Interactive UI built with **Streamlit**
- Real-time predictions
- Modular ML pipeline (preprocessing → training → inference)
- Containerized using **Docker**

---

## 🧠 Tech Stack

- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn (Naive Bayes), PyTorch (Neural Network)  
- **Frontend:** Streamlit  
- **Model Persistence:** Joblib, PyTorch (.pth)  
- **Deployment:** GitHub, Streamlit Cloud  
- **Containerization:** Docker  

---

## 📊 Project Workflow

1. **Data Preprocessing**
   - Handling categorical variables using encoding
   - Feature selection based on correlation
   - Feature scaling using StandardScaler

2. **Model Training**
   - Naive Bayes for baseline performance
   - Neural Network for capturing complex patterns

3. **Model Evaluation**
   - Accuracy and classification metrics comparison

4. **Deployment**
   - Integrated model into Streamlit UI
   - Deployed on Streamlit Cloud

---

## 📁 Project Structure
employee-attrition-ml/
│
├── app/
│ └── streamlit_app.py
├── src/
│ ├── preprocessing.py
│ ├── train_nb.py
│ ├── train_nn.py
│ └── model.py
├── models/
│ ├── nb_model.pkl
│ ├── nn_model.pth
│ ├── scaler.pkl
│ ├── feature_names.pkl
│ └── input_size.pkl
├── data/
│ └── employee.csv
├── requirements.txt
└── README.md



---

## ⚙️ Installation & Setup

### 🔹 Clone Repository

```bash
git clone https://github.com/your-username/employee-attrition-ml.git
cd employee-attrition-ml



Install Dependencies
pip install -r requirements.txt



Run Application
streamlit run app/streamlit_app.py

Docker Setup
Build Image
docker build -t attrition-app .
Run Container
docker run -p 8501:8501 attrition-app

🌐 Deployment

The application is deployed using Streamlit Cloud.

🧠 Model Details
🔹 Naive Bayes
Fast and efficient baseline model
Works well with structured data
🔹 Neural Network
Built using PyTorch
Captures complex patterns in employee behavior
📌 Key Learnings
Importance of feature selection and preprocessing
Handling class imbalance in datasets
Model comparison and evaluation
Deploying ML models into real-world applications
Containerization using Docker



---

# 🔥 WHAT THIS README DOES

✔ Professional  
✔ Recruiter-friendly  
✔ Covers ML + Deployment  
✔ GitHub-ready  

---

# 🚀 NEXT (HIGH VALUE)

If you want to make it even stronger:

👉 Add **project screenshots in README**  
👉 Add **live Streamlit link**  
👉 Add **accuracy comparison table**

---

Just say:

👉 **“add screenshots section”**  
👉 **“improve README for placements”**

I’ll upgrade it to **top-tier portfolio level 🚀🔥**
