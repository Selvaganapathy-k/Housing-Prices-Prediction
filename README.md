Subject: README.md – Housing Prices Prediction

# 🏠 Housing Prices Prediction

## 📌 Project Description

This project implements a **Housing Prices Prediction system** using Machine Learning.
The model predicts the **median house value** based on location, population, housing characteristics, and proximity to the ocean.

The project includes:

* Data analysis and preprocessing
* Model training using **XGBoost Regressor**
* A **Streamlit web application** for real-time house price prediction

This project is developed as a **mini project** for academic learning and practical exposure to regression models and ML deployment.

---

## 📁 Dataset Information

* **Dataset Name:** California Housing Dataset
* **File:** `housing.csv`

The dataset contains attributes such as:

* Longitude and latitude
* Housing median age
* Total rooms and bedrooms
* Population and households
* Median income
* Ocean proximity
* Median house value (target variable)

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit

---

## 📂 Project Structure

```
Housing-Prices-Prediction
│
├── housing.csv
├── housing_price.ipynb
├── xgb_house_model.pkl
├── house_model_columns.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Selvaganapathy-k/Housing-Prices-Prediction
cd Housing-Prices-Prediction
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🌐 Live Application

🔗 **Streamlit App URL:**
[https://housing-prices-prediction-ganapathy.streamlit.app/](https://housing-prices-prediction-ganapathy.streamlit.app/)

---

## 🔍 Model Details

* Problem Type: **Regression**
* Algorithm Used: **XGBoost Regressor**
* Preprocessing:

  * Numerical feature scaling
  * One-hot encoding for ocean proximity
* Input Features:

  * Location details
  * Housing and population statistics
  * Median income
* Output:

  * Predicted median house value (USD)

---

## 📈 Features

* Interactive and user-friendly Streamlit interface
* Real-time house price prediction
* Handles categorical and numerical features
* Uses trained model and stored feature columns
* Robust gradient boosting regression model

---

## 🎓 Learning Outcomes

* Understanding regression problems
* Feature engineering and preprocessing
* Training and evaluating advanced ML models
* Saving and loading ML models using Pickle
* Deploying ML applications using Streamlit
* Structuring end-to-end ML projects on GitHub

---

## 📌 Notes

* Virtual environment folders (`venv`, `myvenv`) are not included in the repository.
* All required dependencies are listed in `requirements.txt`.

---

## ✍️ Author

**Selvaganapathy K**
Computer Science Student

---

## 🏁 Conclusion

This project demonstrates how machine learning models can be applied to real estate data to predict housing prices and support data-driven decision-making.
