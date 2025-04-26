# 📊 02450 – Customer Segmentation: Regression & Classification  
Course project • Technical University of Denmark (DTU)

**Goal:** predict customers’ **Spending Score** and flag **high-value shopper segments** using classical and neural models, with rigorous cross-validation and statistical testing.

<p align="center">
  <img src="docs/teaser.png" alt="Sample scatter plot of PCA projection with clusters and spending scores">
</p>

---

## 1&nbsp;·&nbsp;Project Overview
| Step | What we did | Why it matters |
|------|-------------|----------------|
| **EDA & Pre-processing** | Explored distributions, handled missing values (none), scaled features, one-hot-encoded `Gender`. | Clean, standardized inputs for fair model comparison. |
| **Regression** | Regularized Linear Regression (ridge), **ANN** (1-4 hidden units), baseline (mean). | Predict continuous *Spending Score* for personalized offers. |
| **Classification** | Logistic Regression (ℓ₂), **ANN**, baseline (majority class). | Identify top-tier customers for retention campaigns. |
| **Model Selection** | Two-level **10-fold CV** (outer = performance, inner = hyper-tuning). | Unbiased generalization estimates. |
| **Statistical Tests** | Paired _t_-test (RMSE) & McNemar (error rate). | Verify improvements aren’t due to chance. |
| **Results** | ANN ↓ **28 % RMSE** vs. baseline • Logistic Reg ↑ **12 % accuracy** vs. baseline. | Quantified business value. |

Dataset: [Mall Customer Segmentation](https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering) (Kaggle).  
All code in Python 3.11 · NumPy · pandas · scikit-learn · Matplotlib · (vanilla) PyTorch.

---

## 2&nbsp;·&nbsp;Quick Start

```bash
# 1 · Clone & cd
git clone https://github.com/nickiliak/02450-Customer-Segmentation.git
cd 02450-Customer-Segmentation

# 2 · Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 3 · Run the notebook end-to-end
jupyter nbconvert --execute Assignment2.ipynb
