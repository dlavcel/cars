# 🚗 Car Auction Price Prediction

End-to-end data pipeline that predicts car auction prices using web scraping, feature engineering, computer vision, and machine learning models.

---

## 🎯 Problem

Car auction prices are highly variable and difficult to estimate due to:

* Inconsistent vehicle condition
* Missing or noisy data
* Hidden damage factors
* Market-driven price fluctuations

This project aims to:

* Predict auction prices
* Identify key price drivers
* Improve estimation accuracy using real-world data

---

## ⚙️ Solution Overview

The system is built as a full pipeline:

1. Data collection (scraping auction listings)
2. Data cleaning and normalization
3. Feature engineering
4. Computer vision-based condition analysis
5. Machine learning modeling
6. Prediction blending and evaluation

---

## 🕷️ Data Collection Pipeline

The scraping system is designed as a scalable multi-stage pipeline.

### Architecture

The pipeline is split into two main workers:

#### 1. List Worker

* Iterates through auction listing pages
* Extracts structured summary data
* Collects listing URLs for further processing

#### 2. Detail Worker

* Visits individual listing pages
* Extracts detailed attributes
* Collects image URLs

This separation allows:

* Parallel processing
* Better scalability
* Cleaner data flow

---

### 🔐 Access & Configuration

The scraping pipeline requires authenticated access and environment configuration.

* Session setup and credentials must be prepared beforehand
* External configuration (e.g. proxies, accounts) is not included in the repository
* Sensitive data is intentionally excluded

---

## 👁️ Vehicle Condition Analysis (Computer Vision)

After collecting image URLs, a computer vision stage is applied.

### Pipeline

* Detects visible vehicle damage from images
* Extracts damage-related signals
* Aggregates findings into structured features

### Generated Features

* Damage count
* Damage type
* Estimated severity

These features are later used in the price prediction model.

---

## 🔥 Feature Engineering

### Out-of-Fold Target Encoding

A key feature:

* `make_model_year_price_median`

Represents the **median historical price** for a given:

* make
* model
* year

Calculated using **out-of-fold cross-validation**, preventing data leakage.

👉 This became the most important feature across models.

---

### Additional Features

* Mileage
* Engine volume
* Cylinders
* Damage type & severity
* Fuel type
* Transmission
* Drive type

Custom damage scoring was applied to convert categorical damage into numerical values.

---

## 🧠 Modeling Approach

Multiple models were trained and benchmarked:

* Linear Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

---

### 📉 Log + Linear Blending

Two prediction targets were used:

* Linear price
* Log-transformed price

Final prediction:

* 40% linear
* 60% log

This approach:

* Reduces outlier impact
* Stabilizes predictions
* Improves overall accuracy

---

## 📊 Results

**Best model: XGBoost (blended)**

* MAE: ~1337 €
* RMSE: ~2189 €
* R²: ~0.83

### Prediction Accuracy

* 25% within 10% error
* 48% within 20%
* 66% within 30%

👉 The model captures most of the variance while handling noisy auction data.

---

## 🧠 Key Insights

The most important feature:

* `make_model_year_price_median`

This shows that:

👉 Historical pricing patterns dominate price prediction.


---

## ⚠️ Limitations

* Auction prices are highly volatile
* Vehicle condition estimation is imperfect
* Hidden mechanical issues are not captured

---

## 🚀 Business Value

This system can be used to:

* Estimate fair auction prices
* Identify undervalued vehicles
* Support bidding decisions
* Assist in pricing strategy

---

## 🔄 Full Pipeline

1. Configure access
2. Collect listing-level data
3. Collect detail-level data
4. Extract image-based damage features
5. Generate structured features
6. Train models
7. Blend predictions
8. Evaluate results

---

## ▶️ How to Run

### 1. Configure environment

Prepare required credentials and configuration files.

### 2. Run scraping pipeline

```bash
python manager.py --list-workers --detail_workers
```

### 3. Train models

```bash
python regression_final.py
```

### 4. View results

Model performance metrics will be printed in the console.

---

## 🧰 Tech Stack

* Python (pandas, numpy, scikit-learn)
* XGBoost, LightGBM, CatBoost
* SQL
* Computer Vision (YOLO / image processing)
* Web scraping (requests, BeautifulSoup, cloudscraper)

---


## 💡 Key Takeaway

This project demonstrates:

* End-to-end data pipeline design
* Scalable web scraping architecture
* Advanced feature engineering (OOF encoding)
* Integration of computer vision into ML pipeline
* Real-world regression modeling with noisy data

---
