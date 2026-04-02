# 🚗 Car Auction Price Prediction

End-to-end pipeline for predicting car auction prices using multi-stage scraping, tabular feature engineering, computer vision-based damage severity estimation, and regression modeling.

---

## 🎯 Problem

Car auction prices are difficult to estimate because they depend on multiple noisy factors:

* make and model
* vehicle year
* mileage
* engine characteristics
* reported damage
* visible body damage
* market-specific price behavior

The goal of this project is to:

* predict final auction price
* identify the strongest price drivers
* enrich structured auction data with image-based damage severity signals
* compare multiple regression models on real-world noisy data

---

## ⚙️ Solution Overview

The system is built as a full pipeline:

1. account and access preparation
2. auction data scraping
3. listing-level and detail-level data collection
4. image-based damage severity estimation
5. feature engineering
6. regression model training and benchmarking
7. blended prediction and evaluation

---

## 🕷️ Data Collection Pipeline

The scraping layer is organized as a multi-stage workflow.

### 1. Access Preparation

Before scraping, authenticated access must be prepared.

A separate registration/setup stage is used before the main scraping pipeline starts. External credentials, proxy configuration, and environment-specific files are intentionally excluded from the repository.

### 2. List Worker

The list worker:

* iterates through auction result pages
* extracts summary-level listing data
* stores listing URLs for downstream processing

### 3. Detail Worker

The detail worker:

* visits each listing page
* extracts missing structured attributes
* collects image URLs for visual analysis

This separation allows the scraping pipeline to run in parallel and scale better.

---

## 👁️ Computer Vision Damage Severity Estimation

The project includes a computer vision stage that estimates visual damage severity from vehicle images.

### CV Pipeline

For each vehicle folder:

1. images are loaded from the vehicle directory
2. YOLO models detect visible damage on each image
3. detections are converted into view-level severity scores
4. severity is aggregated into final damage features
5. these final features are added to the regression dataset

### Primary and Secondary Damage

The pipeline evaluates two damage channels separately:

* **primary damage severity**
* **secondary damage severity**

Two YOLO models are used independently:

* one model for **primary damage**
* one model for **secondary damage**

Each model has its own inference parameters such as:

* image size
* confidence threshold
* tile IoU
* merge IoU
* overlap

### Tiled Inference

To improve damage detection on large vehicle images, the pipeline uses tiled prediction:

* the image is split into overlapping tiles
* YOLO runs on each tile
* detections are merged using class-aware NMS

This helps recover smaller or localized damage regions that could be missed in a single-pass prediction.

### View-Based Severity Scoring

Each image is treated as a specific vehicle view.

Depending on the auction source, the pipeline supports either:

* 4-view layout
* 6-view layout

Detected damage is converted into a severity score for each view based on:

* damage class
* confidence
* bounding box extent
* box count
* critical damage types
* soft confidence-based boosts

### Damage Classes

The severity logic uses detection classes such as:

* scratch
* rub
* dent
* crack
* lamp broken
* tire flat
* dislocated part
* no part
* glass shatter
* crash

More severe classes have larger weights in the scoring function.

### Final CV Features

The CV stage produces final structured signals such as:

* `damage_primary_severity`
* `damage_secondary_severity`

These are not stored as raw detection outputs in the regression table.
Instead, detections are processed immediately, transformed into severity scores, and only the final aggregated features are added to the modeling dataset.

---

## 🔥 Feature Engineering

The tabular pipeline includes:

* text normalization
* missing value handling
* currency conversion
* rare category grouping
* custom damage severity transformation
* historical price aggregation features

### Out-of-Fold Historical Price Feature

A key engineered feature is:

* `make_model_year_price_median`

This feature represents the median historical price for a given:

* make
* model
* year

It is computed in an out-of-fold way to reduce target leakage during training.

This became one of the strongest predictors across the final models.

### Structured Features Used for Regression

Examples of features used in the regression pipeline:

* make
* model
* fuel type
* transmission
* drive type
* mileage
* engine volume
* cylinders
* year
* primary damage
* secondary damage
* `damage_primary_severity`
* `damage_secondary_severity`
* `make_model_year_price_median`

---

## 🧠 Modeling Approach

Several regression models were trained and benchmarked:

* Linear Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Linear + Log Target Strategy

Two target variants were used:

* raw price
* log-transformed price

Final prediction is a weighted blend of both:

* 40% linear prediction
* 60% log-based prediction

This makes predictions more stable and reduces sensitivity to extreme price values.

---

## 📊 Results

**Best model: XGBoost (blended)**

* MAE: ~1337
* RMSE: ~2189
* R²: ~0.83

### Prediction Accuracy

* 25% of predictions within 10% error
* 48% within 20% error
* 66% within 30% error

These results show that the model captures a large share of auction price variance despite noisy and incomplete real-world data.

---

## 🧠 Key Insights

The strongest predictive signal was:

* `make_model_year_price_median`

This suggests that historical price behavior for a specific make-model-year combination is highly informative.

Other important signals included:

* mileage
* reported primary damage
* engine volume
* year
* model / make
* image-based damage severity

---

## ⚠️ Limitations

* auction prices are highly volatile
* some hidden mechanical issues cannot be inferred from images
* image coverage is limited to available listing photos
* CV severity depends on detection quality and image viewpoint
* structured damage labels from source data may be incomplete or noisy

---

## 🚀 Business Value

This system can help:

* estimate fair auction prices
* support bidding decisions
* identify potentially undervalued vehicles
* enrich auction listings with visual damage severity
* analyze which structured and visual signals affect price the most

---

## 🔄 Full Pipeline

1. prepare access and configuration
2. run account/session setup
3. scrape listing-level auction data
4. scrape detail-level information and image URLs
5. run YOLO-based image analysis
6. convert detections into damage severity features
7. build regression dataset
8. train and compare models
9. blend predictions
10. evaluate model quality

---

## ▶️ How to Run

### 1. Prepare environment

Configure required credentials, external files, and environment-specific settings.

### 2. Prepare access

```bash
python register_accounts.py proxies.txt accounts_ready.txt yourdomain.com
```

### 3. Run scraping manager

```bash
python manager.py --list-worker --detail-worker
```

### 4. Run CV severity estimation

```bash
python evaluate_primary_secondary.py
```

### 5. Train and benchmark regression models

```bash
python regression_final.py
```

---

## 🧰 Tech Stack

* Python
* pandas
* numpy
* scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Ultralytics YOLO
* PyTorch
* OpenCV
* BeautifulSoup
* cloudscraper
* SQL

---

---

## 💡 Key Takeaway

This project demonstrates:

* multi-stage scraping architecture
* integration of computer vision into tabular ML pipeline
* custom image-to-severity feature generation
* leakage-aware feature engineering
* regression model benchmarking
* blended prediction strategy for real-world auction data
