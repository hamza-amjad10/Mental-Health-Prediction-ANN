# Mental Health Prediction using Neural Networks

This project uses machine learning (Neural Networks) to predict whether an individual is likely to seek mental health treatment based on survey data.

## Dataset
The dataset is obtained from [Kaggle: Mental Health Dataset](https://www.kaggle.com/datasets/divaniazzahra/mental-health-dataset) and includes features such as:

- Gender
- Country
- Occupation
- Self-employed status
- Family mental health history
- Mood swings, stress, coping strategies, etc.

## Approach
1. **Data Cleaning & Preprocessing**
   - Missing values filled
   - Categorical variables encoded using mapping, one-hot encoding, and ordinal encoding
2. **Feature Selection**
   - Target: `treatment`
   - Input features: 27 preprocessed features
3. **Model**
   - Neural Network with 3 hidden layers (64 → 128 → 256 neurons)
   - Activation: ReLU for hidden layers, Sigmoid for output
   - Optimizer: Adam with learning rate 0.0005
   - Loss: Binary Crossentropy
4. **Training**
   - 20 epochs, batch size 64
   - Validation split: 0.2

## Results
- **Accuracy:** 0.77
- **Confusion Matrix:**
[[20551 8344]
[ 5180 24398]]

- **Classification Report:**
           precision    recall  f1-score   support

       0       0.80      0.71      0.75     28895
       1       0.75      0.82      0.78     29578

accuracy                           0.77     58473


## Possible Improvements
- Experiment with deeper or different neural network architectures
- Feature engineering to extract more meaningful insights
- Hyperparameter tuning (learning rate, batch size, epochs)
- Trying other models like Random Forest, XGBoost, or Gradient Boosting
- Addressing class imbalance if present

## How to Run
1. Clone the repo
2. Download `Mental Health Dataset` from Kaggle
3. Place `kaggle.json` in root directory
4. Run the notebook `Mental_Health_Prediction.ipynb` in Google Colab or locally with necessary Python libraries:
 - pandas, numpy, matplotlib
 - scikit-learn
 - tensorflow / keras

---

