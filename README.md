# IBM Machine Learning Lab Codes

A collection of hands-on machine learning scripts I worked through and built up during the **IBM AI Engineering / Machine Learning** course. Think of it as my personal notebook — every classic ML topic from the curriculum is here as a self-contained script, each one training a real model on a real dataset and showing the results with plots.

These are the files I wrote while following along with the IBM ML0101EN labs, plus a couple of mini capstone-style projects I came up with on my own to tie everything together.

---

## What's inside

Each script covers one ML concept. Most are fully self-contained — they download their dataset at runtime, so no separate data files are needed.

### Supervised Learning — Regression
| File | What it does |
|------|--------------|
| `MLR.py` | Multiple linear regression to predict **CO₂ emissions** from engine size & fuel consumption |
| `DecisionTreeRegressor.py` | Decision tree regression to predict **NYC taxi tips** (with L1 normalization) |
| `RandomForest.py` | Random forest regression on the **California housing** dataset |
| `RFXGB.py` | Random forest vs. ideal model comparison on California housing |
| `RegressionEval.py` | Full regression evaluation (MAE, MSE, RMSE, R², residuals) on California housing |
| `RegularizationReg.py` | Deep dive into **Ridge vs. Lasso vs. plain Linear** regression — outliers, coefficients, feature selection |

### Supervised Learning — Classification
| File | What it does |
|------|--------------|
| `LogR.py` | Logistic regression to predict **customer churn** |
| `DecisionTree.py` | Decision tree to recommend **(drugs)** based on patient vitals |
| `KNN.py` | K-Nearest Neighbors for **customer categorization** (with K-tuning plot) |
| `SVM.py` | SVM & decision tree on a highly imbalanced **credit card fraud** dataset with class weighting |
| `RandomForest.py` | Random forest regression on the **California housing** dataset |
| `Multiclass-Classification.py` | **OvA vs. OvO** multiclass logistic regression for obesity-level prediction |
| `Titanicsurvivalprediction.py` | Good old **Titanic** survival prediction with pipelines + GridSearchCV (RF & logistic) |
| `ClassificationEval.py` | Evaluating KNN vs. SVM on breast cancer data, with a noise/overfitting experiment |

### Unsupervised Learning
| File | What it does |
|------|--------------|
| `K-means.py` | K-Means clustering on synthetic data **and** customer segmentation |
| `ClusteringEval.py` | Evaluating K-Means with **silhouette** & **Davies-Bouldin** scores |
| `PCA.py` | Principal Component Analysis — synthetic data and the classic Iris 2D reduction |
| `TSne.py` | t-SNE visualization (3D blobs projected to 2D) |

### Pipelines & Capstone
| File | What it does |
|------|--------------|
| `ML_Pipelines_and_GridSearchCV.py` | StandardScaler → PCA → KNN pipeline with GridSearchCV hyperparameter tuning |
| `FinalProject.py` | My mini capstone: **predicting rain in Melbourne** from 2008–2017 weather data, with feature importance across two models |

---

## Getting started

### 1. Clone the repo
```bash
git clone https://github.com/Invictus596/IBM-Codes
cd IBM-Codes
```

### 2. Install the dependencies
```bash
pip install -r requirements.txt
```

### 3. Run any script
```bash
python FinalProject.py
```

Every script prints its evaluation metrics to the console and pops up one or more matplotlib plots, so you can see exactly what the model is doing.

---

## Requirements

- **Python 3.8+**
- All third-party libraries are listed in [`requirements.txt`](requirements.txt)
- An internet connection (some scripts pull datasets from IBM's cloud at runtime)

> **Note:** `SVM.py` reads a local file path (`/home/aziz/Downloads/creditcard.csv`) instead of a URL. You'll need to download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and update that path before running it.

---

## A note on the code

These scripts were written as I was learning, so they're a bit rough around the edges — lots of exploration, trial and error, and comments that document what I was thinking at the time. Some won't be perfectly cleaned up like production code, but that's kind of the point: they show the learning journey. Feel free to fork, tidy up, or build on any of them.

Happy modeling!
