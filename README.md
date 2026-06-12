# 🏠 Hyderabad House Price Predictor

> An end-to-end Machine Learning web application that predicts residential property prices in Hyderabad, India using ensemble learning and a Streamlit-powered user interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Real estate pricing in Hyderabad is highly dynamic and influenced by a wide range of factors — from property size and location to amenities like swimming pools, jogging tracks, and lift availability. Traditional price estimation methods rely heavily on human expertise and local market knowledge, making them inconsistent and time-consuming.

This project addresses that challenge by building a **data-driven house price prediction system** trained on real Hyderabad property listings. The system:

- Ingests raw property data with 40 attributes
- Cleans and preprocesses it through a reproducible pipeline
- Trains and compares **10 machine learning models** (Linear, Ridge, Lasso, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM)
- Selects the best-performing model (Random Forest Regressor with 200 estimators)
- Deploys a real-time prediction interface using **Streamlit**

**Target Output:** Predicted property price in Indian Rupees (₹)

---

## ✨ Features

- Exploratory Data Analysis with correlation heatmaps and location-based price visualizations
- Outlier detection and removal using price thresholding
- Missing value treatment via sentinel value (`9`) replacement with `NaN` followed by row-level dropping
- Feature selection from 40 raw columns to 8 most predictive features
- Scikit-Learn ML pipeline with `ColumnTransformer` for automated preprocessing
- Median imputation for numerical features, mode imputation for categorical features
- StandardScaler normalization for linear models; raw features for tree-based models
- OneHotEncoding for the `Location` categorical feature (239 unique locations)
- Hyperparameter tuning using GridSearchCV for Ridge and Lasso (5-fold cross-validation)
- Comparison of 10 models with R² Score as the evaluation metric
- Model serialization using `joblib`
- Interactive Streamlit web app with real-time price prediction
- Two-column responsive UI layout with dynamic location dropdown

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations and NaN handling |
| Scikit-Learn | ML pipelines, preprocessing, model training, evaluation |
| Matplotlib | Scatter plots, bar charts, EDA visualizations |
| Seaborn | Correlation heatmap, regression plots |
| Streamlit | Web application and prediction UI |
| Joblib | Model serialization and deserialization |
| XGBoost | Extreme Gradient Boosting regressor |
| CatBoost | Gradient boosting with categorical feature support |
| LightGBM | Fast gradient boosting framework |
| Git / GitHub | Version control and repository hosting |

---

## 🔄 Project Workflow

```
Raw Dataset (Hyderabad.csv)
           ↓
   Data Loading & Inspection
   (shape, dtypes, isnull checks)
           ↓
   Sentinel Value Treatment
   (replace 9 → NaN)
           ↓
   Missing Value Removal
   (dropna on affected rows)
           ↓
   Outlier Removal
   (Price > ₹10 Crore filtered out)
           ↓
   Exploratory Data Analysis
   (correlation matrix, scatter plots,
    location-based price analysis)
           ↓
   Feature Selection
   (8 key features chosen from 40)
           ↓
   Cleaned Dataset Exported
   (Hyderabad_houses_cleaned_DS.csv)
           ↓
   Train-Test Split (80/20, seed=42)
           ↓
   Preprocessing Pipeline
   (Imputation → Scaling → OHE)
           ↓
   Model Training & Comparison
   (10 models: Linear → Ensemble)
           ↓
   Best Model Selection
   (Random Forest Regressor)
           ↓
   Model Serialization
   (joblib → house_price_model.pkl)
           ↓
   Streamlit Deployment
   (app.py → real-time UI)
           ↓
   User Prediction
   (Input → Model → ₹ Output)
```

---

## 📊 Dataset Description

**File:** `Hyderabad.csv` (raw) → `Hyderabad_houses_cleaned_DS.csv` (cleaned)
**Records:** 2,517 raw → 2,433 after cleaning
**Features:** 40 columns total

### Feature Table

| Feature | Type | Description |
|---|---|---|
| **Price** | int64 | Target variable — property price in INR (₹) |
| **Area** | int64 | Total property area in square feet |
| **Location** | string | Locality/neighborhood name in Hyderabad (239 unique) |
| **No. of Bedrooms** | int64 | Number of bedrooms (1–8) |
| **Resale** | binary (0/1) | Whether the property is a resale unit |
| **MaintenanceStaff** | binary (0/1) | Maintenance staff available |
| **Gymnasium** | binary (0/1) | Gymnasium facility present |
| **SwimmingPool** | binary (0/1) | Swimming pool available |
| **LandscapedGardens** | binary (0/1) | Landscaped garden present |
| **JoggingTrack** | binary (0/1) | Jogging track available |
| **RainWaterHarvesting** | binary (0/1) | Rainwater harvesting system present |
| **IndoorGames** | binary (0/1) | Indoor games facility |
| **ShoppingMall** | binary (0/1) | Shopping mall nearby/on premises |
| **Intercom** | binary (0/1) | Intercom facility |
| **SportsFacility** | binary (0/1) | Sports facility available |
| **ATM** | binary (0/1) | ATM on premises |
| **ClubHouse** | binary (0/1) | Clubhouse available |
| **School** | binary (0/1) | School nearby/on premises |
| **24X7Security** | binary (0/1) | 24x7 security available |
| **PowerBackup** | binary (0/1) | Power backup available |
| **CarParking** | binary (0/1) | Car parking facility |
| **StaffQuarter** | binary (0/1) | Staff quarters available |
| **Cafeteria** | binary (0/1) | Cafeteria on premises |
| **MultipurposeRoom** | binary (0/1) | Multipurpose room available |
| **Hospital** | binary (0/1) | Hospital nearby |
| **WashingMachine** | binary (0/1) | Washing machine included |
| **Gasconnection** | binary (0/1) | Gas connection available |
| **AC** | binary (0/1) | Air conditioning installed |
| **Wifi** | binary (0/1) | Wi-Fi facility available |
| **Children'splayarea** | binary (0/1) | Children's play area present |
| **LiftAvailable** | binary (0/1) | Elevator/lift in building |
| **BED** | binary (0/1) | Bed included with property |
| **VaastuCompliant** | binary (0/1) | Vaastu compliant property |
| **Microwave** | binary (0/1) | Microwave included |
| **GolfCourse** | binary (0/1) | Golf course access |
| **TV** | binary (0/1) | Television included |
| **DiningTable** | binary (0/1) | Dining table included |
| **Sofa** | binary (0/1) | Sofa included |
| **Wardrobe** | binary (0/1) | Wardrobe included |
| **Refrigerator** | binary (0/1) | Refrigerator included |

---

## 🧹 Data Preprocessing

### Step 1: Sentinel Value Treatment
The raw dataset uses `9` as a placeholder for missing/unreported amenity values. All occurrences of `9` across amenity binary columns were replaced with `NaN`:
```python
data.replace(9, np.nan, inplace=True)
```

### Step 2: Missing Value Removal
After NaN conversion, rows containing any missing values were dropped:
```python
data.dropna(inplace=True)
```

### Step 3: Outlier Removal
Properties priced above ₹10 Crore (₹100,000,000) were identified as statistical outliers and removed to reduce skewness and improve model generalization:
```python
data = data[data['Price'] <= 100000000]
```

### Step 4: Feature Selection
From 40 raw columns, 8 most impactful features were selected based on correlation analysis and domain relevance for the production model. The cleaned dataset was exported as `Hyderabad_houses_cleaned_DS.csv`.

### Step 5: Scikit-Learn Preprocessing Pipeline

For tree-based models (production), a `ColumnTransformer` pipeline handles:

**Numerical features** (`Area`, `No. of Bedrooms`, `Resale`, `CarParking`, `SwimmingPool`, `JoggingTrack`, `LiftAvailable`):
- `SimpleImputer(strategy="median")` — fills any remaining gaps with median value
- `StandardScaler()` — zero-mean, unit-variance normalization

**Categorical features** (`Location`):
- `SimpleImputer(strategy="most_frequent")` — fills with mode
- `OneHotEncoder(handle_unknown="ignore")` — creates binary columns per location

---

## 📈 Exploratory Data Analysis

The notebook `preprocessing/house_pp.ipynb` includes:

- **Descriptive Statistics:** `data.describe()` on all 40 columns
- **Correlation Heatmap:** Full 39×39 numeric correlation matrix (`corr_matrix.png`) using Seaborn heatmap to identify top predictors of `Price`
- **Scatter Plot (Price vs Area):** Identifies the positive linear trend between property size and price, with outlier separation using color-coded hue
- **Regression Plot:** Seaborn `regplot` overlaying trend line on Price vs Area scatter
- **Location-Based Price Analysis (`cation_analysis_crores.png`):** Dual horizontal bar chart showing Top 15 most expensive and Top 15 most affordable Hyderabad localities by average property price (in ₹ Crores)
- **Amenity Distribution:** `value_counts()` analysis on all binary amenity features to understand prevalence across the dataset

**Key Insights:**
- Area has the strongest positive correlation with Price among all numeric features
- Premium localities (e.g., Hitech City, Gachibowli, Kokapet) command 3–5× higher prices than affordable areas
- Most amenity features are binary and sparse; only a subset (CarParking, LiftAvailable, SwimmingPool, JoggingTrack) showed meaningful correlation with price

---

## ⚙️ Feature Engineering

No synthetic feature creation was performed. Feature engineering primarily consisted of:

- **Amenity Encoding:** All binary amenity features were already numerically encoded (0/1) in the raw dataset
- **Location Encoding:** For linear models in the notebook, `pd.get_dummies()` was applied to `Location` (drop_first=True). For the production pipeline in `train_model.py`, `OneHotEncoder` inside the Scikit-Learn pipeline handles this automatically
- **Feature Subsetting:** 40 raw columns reduced to 8 selected features for the production model

---

## 🤖 Model Training

### Train-Test Split
- **Split ratio:** 80% training / 20% testing
- **Random state:** 42 (reproducibility)

### Phase 1: Linear Models (Notebook EDA)

| Model | Configuration |
|---|---|
| Linear Regression | Default; trained on StandardScaler-normalized OHE features |
| Ridge Regression | GridSearchCV over alpha=[0.001, 0.01, 0.1, 1, 10, 50, 100, 500], 5-fold CV |
| Lasso Regression | GridSearchCV over alpha=[0.0001, 0.001, 0.01, 0.1, 1, 10, 50], 5-fold CV, max_iter=10000 |

### Phase 2: Ensemble Models (Notebook + Production)

| Model | Configuration |
|---|---|
| Random Forest | n_estimators=200, random_state=42 |
| Extra Trees | n_estimators=200, random_state=42 |
| AdaBoost | n_estimators=200, random_state=42 |
| Gradient Boosting | n_estimators=300, learning_rate=0.05, random_state=42 |
| XGBoost | n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42 |
| CatBoost | iterations=300, learning_rate=0.05, depth=6, verbose=0 |
| LightGBM | n_estimators=300, learning_rate=0.05, random_state=42 |

---

## 📐 Evaluation Metrics

All models were evaluated using **R² Score (Coefficient of Determination)**:

```
R² = 1 - (SS_res / SS_tot)
```

where `SS_res` is the residual sum of squares and `SS_tot` is the total sum of squares.

An R² of 1.0 indicates perfect predictions; 0 means the model is no better than the mean baseline.

---

## 🚀 Deployment

### Model Saving
The best-performing pipeline (preprocessing + Random Forest) is serialized with `joblib`:
```python
joblib.dump(best_rf_model, "house_price_model.pkl")
```

### Streamlit Application (`app.py`)
The app loads the model and cleaned dataset at startup:
```python
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("Hyderabad_houses_cleaned_DS.csv")
```

**User Inputs (two-column layout):**

| Column 1 | Column 2 |
|---|---|
| Area (sq.ft) — number input | Car Parking — Yes/No dropdown |
| Location — searchable dropdown (239 options) | Swimming Pool — Yes/No dropdown |
| No. of Bedrooms — number input | Jogging Track — Yes/No dropdown |
| Resale Property — Yes/No dropdown | Lift Available — Yes/No dropdown |

**Prediction Flow:**
1. User fills inputs and clicks **"🔮 Predict House Price"**
2. Input assembled as a single-row `pd.DataFrame`
3. `model.predict(input_df)` runs the full preprocessing pipeline + Random Forest inference
4. Predicted price displayed as: `🏠 Estimated House Price: ₹ X,XX,XX,XXX.XX`
5. Property summary table rendered below the prediction

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/vnareshrajput/House_Price_Predictor-Hyderabad.git
cd House_Price_Predictor-Hyderabad

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

> The pre-trained model file `house_price_model.pkl` is included in the repository. You can also retrain it by running:
> ```bash
> python train_model.py
> ```

---

## 🔮 Future Improvements

- Add more evaluation metrics (MAE, RMSE, MAPE) to the training pipeline for richer model diagnostics
- Implement hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest production model
- Expand feature set to include floor number, property age, and proximity to metro stations
- Add geospatial visualization (map view) to show predicted prices by location on a Hyderabad map
- Incorporate SHAP (SHapley Additive exPlanations) for model interpretability and feature importance visualization
- Implement CI/CD pipeline for automated model retraining with new property data
- Deploy on Streamlit Cloud or Hugging Face Spaces for public access
- Add confidence interval or price range output instead of a single-point estimate

---

## 👨‍💻 Author

**V. Naresh Rajput**
Final Year Student — Data Science & Analytics
Indian Institute of Technology (IIT) Roorkee

- 📧 naresh_v@bt.iitr.ac.in
- 🔗 [GitHub: vnareshrajput](https://github.com/vnareshrajput)
- 🏫 IIT Roorkee, Uttarakhand, India

---

## 📁 Repository Structure

```
House_Price_Predictor-Hyderabad/
│
├── app.py                              # Streamlit web application
├── train_model.py                      # Model training and serialization script
├── requirements.txt                    # Python dependencies
├── house_price_model.pkl               # Serialized Random Forest pipeline (24 MB)
├── Hyderabad_houses_cleaned_DS.csv     # Cleaned dataset used by app and model
│
└── preprocessing/
    ├── Hyderabad.csv                   # Raw dataset (2,517 records, 40 features)
    ├── Hyderabad_houses_cleaned_DS.csv # Cleaned dataset (mirror copy)
    ├── house_pp.ipynb                  # Main EDA + preprocessing + model comparison notebook
    ├── test.ipynb                      # Pipeline testing and validation notebook
    ├── corr_matrix.png                 # Full correlation heatmap (39×39)
    └── cation_analysis_crores.png      # Location-based price analysis chart
```

---

*Built with using Python, Scikit-Learn, and Streamlit*
