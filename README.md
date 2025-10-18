# 🧠 Multimodal Price Prediction 

A machine learning solution developed for the **Amazon ML Challenge**, combining **tabular**, **text**, and **image** features to predict product prices using a **multimodal stacking ensemble**.  
The approach integrates **LightGBM**, **Sentence Transformers**, and **CNN-based embeddings** for robust performance across diverse data modalities.

---

## 🚀 Executive Summary

Our solution employs a **multimodal stacking ensemble** designed to predict product prices by synergistically combining signals from tabular, text, and image data.  
We developed three specialist base models to independently analyze each data modality, and used their predictions as meta-features for a final **LightGBM** model.  

This hierarchical "3+1" approach allowed the final model to learn from the relative strengths and weaknesses of its components, achieving a **robust and highly accurate result**.

---

## ⚙️ Tech Stack

- **Languages:** Python, NumPy, Pandas  
- **Models:** LightGBM, Ridge Regression, EfficientNetB0  
- **Libraries:** scikit-learn, Sentence Transformers, Optuna, TensorFlow/Keras  
- **Visualization:** Matplotlib, Seaborn  

---

## 🧩 Methodology Overview

### 🔹 Problem Analysis

Our initial **Exploratory Data Analysis (EDA)** revealed a complex, multimodal regression problem.  
The price (our target variable) exhibited a significant **right-skew**, necessitating a **logarithmic transformation (`log1p`)** to create a more stable target for modeling.

**Key Observations:**
- **Dominance of Quantitative Features:** The strongest predictors of price were quantitative, such as `normalized_quantity` and `pack_multiplier`.  
- **Contextual Brand Value:** A brand's value was highly contextual. The derived feature `brand_price_density` (average price per unit of quantity per brand) proved far more predictive than brand name alone.  
- **Latent Signals in Unstructured Data:**  
  - Text descriptions contained semantic indicators like “organic” or “gourmet,” signaling premium products.  
  - Images revealed packaging quality, correlating with price.

---

### 🔹 Solution Strategy

Based on these findings, we adopted a **Multimodal Stacking Ensemble** strategy our **“3+1” architecture** designed to maximize signal extraction from each data type while mitigating the weaknesses of individual models.

**Approach Type:** Stacking Ensemble (Multimodal)  
**Core Innovation:**  
Three diverse specialist models (Tabular, NLP, and Vision) generate **out-of-fold predictions** that serve as **meta-features** for a final **LightGBM** meta-learner.  
This enables the ensemble to learn complex, conditional relationships and effectively correct its components’ errors.

---

## 🏗️ Model Architecture

### Architecture Overview

Our "3+1" architecture is a two-level learning system:  
- **Level 0:** Three independent specialist models (Tabular, Text, Vision).  
- **Level 1:** A single **LightGBM meta-learner** combining their predictions.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1-u8bnYOV3gfaUsd__adFMs4B7PuCFmW3" alt="Model Architecture" width="400"/>
</p>

---

### Model Components

#### 🧮 The Tabular Workhorse
- **Goal:** Achieve maximum accuracy using structured data.  
- **Model Type:** LightGBM Regressor  
- **Key Features:** `normalized_quantity`, boolean flags (`is_organic`, etc.), one-hot encoded categorical variables.  
- **Advanced Feature Engineering:**
  - Cross-validated target encoding for `brand`.  
  - `brand_price_density`: average price per unit of normalized quantity (computed safely within a CV loop).  
  - Brand-level statistics (mean, std, max price) and brand-unit interaction terms.  
- **Preprocessing:** Target transformed via `np.log1p()`; numeric features scaled with `RobustScaler`.  
- **Hyperparameters:** Tuned using **Optuna**, with a **Huber loss** objective for robustness to outliers.

#### 💬 The Text Specialist
- **Goal:** Predict price using semantic understanding of product text.  
- **Preprocessing:**
  - Converted `text_for_embedding` using **all-MiniLM-L6-v2 Sentence Transformer** → 384-dimensional embeddings.  
  - Embeddings precomputed and cached for efficiency.  
- **Model Type:** LightGBM Regressor (upgraded from Ridge for non-linear modeling).  
- **Hyperparameters:** Tuned via a targeted Optuna search for optimal embedding performance.

#### 🖼️ The Vision Specialist
- **Goal:** Predict price using visual cues from product images.  
- **Preprocessing:**
  - Images downloaded from `image_link`, resized to **224×224**, and normalized.  
- **Model Type:**
  - **EfficientNetB0** (pretrained CNN) as a feature extractor → 1280-d image embeddings.  
  - **Ridge Regression** trained on embeddings to predict `price_log1p`.

---

## 📊 Model Performance

Model evaluation was performed via **5-fold cross-validation**, reporting **SMAPE** based on combined out-of-fold predictions — ensuring reliable generalization estimates.

### Validation Results

| Model | Type | SMAPE |
|--------|-------|--------|
| Tabular | LightGBM | 55.12 |
| Text | LightGBM on Sentence Embeddings | 53.44 |
| Vision | Ridge on EfficientNet | 57.00 |
| **Ensemble (3+1)** | LightGBM Meta | **50.84** |

---

## 🧠 Conclusion

Our **“3+1” multimodal ensemble** proved to be a highly effective strategy for this complex pricing challenge.  
The approach demonstrated that combining diverse, specialized models can outperform any single model by leveraging complementary strengths.

> **Key Takeaway:**  
> The synergy of the ensemble is greater than the sum of its parts — enabling a more robust and accurate final prediction model.

---

## 📁 Appendix

### A. Code Artifacts
📂 Drive Link:  
[Amazon ML Challenge Artifacts](https://drive.google.com/drive/folders/1xz1JGLxjwwm8cC00YD_m9q_zO7jfVodj?usp=sharing)

📂 Repo Link: 
[Repository Link](https://github.com/Parth-Singla-123/PriceSenseAI)

---

### B. Additional Results

#### Base Model Feature Importance
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1kNrn2fehYnEbXvj8eN89tHxq0o0xHEa1" alt="Feature Plot" width="400"/>
</p>

#### Ensemble Feature Importance
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1IQAd-IGO7pArkvlAXeJa1MCVZ0hPeWe8" alt="Final Ensemble Dependency" width="400"/>
</p>

---

## 🙏 Acknowledgements
Developed as part of the **Amazon ML Challenge**.  
Special thanks to the Amazon dataset team and the open-source ML community for providing valuable tools and pretrained models that made this work possible.
