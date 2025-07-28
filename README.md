# 🌐 FuturePathFinder

**FuturePathFinder** is a career recommendation system that uses a **Random Forest Classifier** to analyze student data and suggest potential career paths. The project includes data cleaning, feature importance analysis, and a sleek **FastAPI-based web interface** for interaction.

---

## 📁 Project Structure

```
FuturePathFinder/
├── main.py
├── data/
│   ├── data_clean.py
│   ├── datasets/
│   │   └── cleaned_data.csv
│   └── data_clean_visualizations/
├── feature/
│   ├── feature_imp.py
│   ├── random_forest_model.pkl
│   └── feature_visualizations/
├── templates/
│   ├── index.html
│   ├── index.js
│   └── style.css
└── README.md
```

---

## 🧼 Data Cleaning (`data/data_clean.py`)

**Goal:** Prepare the raw dataset for model training.

### ✔️ Tasks

- Drop irrelevant columns (e.g., `Student_ID`)
- Handle missing data (median for numeric, mode for categorical)
- Label encode categorical columns (Gender, Job Level, etc.)
- Save cleaned data to `data/datasets/cleaned_data.csv`
- Generate visualizations of missing data and distributions to `data/data_clean_visualizations/`

---

## 🔍 Feature Importance (`feature/feature_imp.py`)

**Goal:** Analyze which features are most predictive of a student’s career.

### ✔️ Tasks

- Load cleaned data
- Train a Random Forest Classifier
- Compute and plot:
  - 📊 Feature importance bar chart  
  - ♨️ Correlation heatmap  
  - 🎯 Target distribution  
  - 🔍 Pairplot of top features  
  - 📈 Cumulative importance plot  
- Save trained model as `random_forest_model.pkl`

### 📁 Output

Visualizations saved in `feature/feature_visualizations/`

---

## 🌐 Web Interface

### 📌 `main.py`

Runs a **FastAPI** app that:

- Serves `index.html` from the `/` route
- Accepts form input and predicts a career using the trained model

### 📁 `templates/`

- `index.html`: Input form and display UI
- `style.css`: Modern, responsive styling (glassmorphism, gradients, dark mode)
- `index.js`: *(Optional)* Dynamic UI handling and fetch API integration

---

## ▶️ Running the App

```bash
uvicorn main:app --reload
```

Then open your browser at: [http://localhost:8000](http://localhost:8000)

---

## ⚙️ Installation & Setup

### 🔧 Requirements

- Python 3.7+
- `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `missingno`, `fastapi`, `jinja2`, `uvicorn`, `joblib`

### ✅ Install all dependencies

```bash
pip install -r requirements.txt
```

### 📦 Sample `requirements.txt`

```
fastapi
jinja2
uvicorn
pandas
numpy
scikit-learn
matplotlib
seaborn
missingno
joblib
```

---

## 📊 Visualizations Overview

| Type                        | Description                              |
|----------------------------|------------------------------------------|
| 🧱 Missing Matrix/Dendrogram | Pre/post missing data views               |
| 📉 Boxplots & Violinplots   | Imputation effect comparisons             |
| 🎯 Target Distribution      | Output class balance                      |
| 📊 Feature Importance       | Ranked bar chart                          |
| ♨️ Correlation Heatmap      | Feature inter-dependence                  |
| 🔍 Pairplot of Top Features | Relationships between top factors         |
| 📈 Cumulative Importance     | How much variance top features explain    |

---

## 🧠 How It Works

- **Input**: User provides data like gender, GPA, job level, etc.
- **Prediction**: FastAPI backend loads `random_forest_model.pkl`, transforms input, and predicts the career.
- **Output**: Recommended professional career is displayed with explanations or confidence (if extended).

---

## 🧪 Customization

- 🔁 Modify the dataset or add new features
- 🤖 Swap out Random Forest with other models (XGBoost, SVM, etc.)
- 🎨 Update `style.css` or `index.html` for a unique frontend design
- 🧩 Extend backend with logging, user sessions, or database storage

---

## 👨‍💻 Author

**Dikshanta Chapagain**  
*Passionate about transforming raw data into meaningful career recommendations.*

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.
