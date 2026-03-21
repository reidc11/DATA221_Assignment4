#Question 1
from sklearn.datasets import load_breast_cancer
import pandas as pd

#Construct the feature matrix X and target vector y.
breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

#Report the shape of X and y
print(feature_matrix.shape)
print(target_vector.shape)

#The feature matrix X has a shape of (569, 30), representing 569 samples and 30 features,
# while the target vector y has a shape of (569,), containing one label per sample.

#Report the number of samples belonging to each class.
print(pd.Series(target_vector).value_counts())

#The target vector contains 357 benign samples (class 1) and 212 malignant samples (class 0),
# for a total of 569 samples

#The dataset is moderately imbalanced, with benign samples (357) outnumbering malignant samples (212)
# at approximately a 60/40 ratio.

#imbalanced classes can make accuracy a misleading metric, and the model may be biased toward predicting
# the majority class