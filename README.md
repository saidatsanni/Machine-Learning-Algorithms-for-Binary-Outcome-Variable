# Machine Learning Algorithms for Binary Outcome Variable Prediction

**Goal**

The goal of this study is to conduct a comprehensive analysis, build a predictive model to predict whether an applicant is considered a good or bad credit risk, and identify the model that provides the best prediction accuracy. Machine Learning methods such as Logistic Regression Models, Variable Selection, Classification Trees, Bagging, Random Forest, Boosted Regression Trees, Generalized Additive Models, and Neural Networks are considered. The model performances are visualized using the ROC Curve and Confusion Matrix. Assessment and selection of the best model is conducted using the Area Under the Curve (AUC), Symmetric Misclassification Rate (MR), and Asymmetric Misclassification Rate (AMR).


**Data**

The German Credit Data contains data on 20 variables and the classification of whether an applicant is considered a Good or Bad credit risk for 1000 loan applicants. The data can be found [here](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

**Statistical Techniques**

* Generalized Linear Model (Logistic Regression Analysis)
* Variable and Model Selection 
* Classification Trees
* Bagging
* Random Forest
* Boosted Regression Trees
* Generalized Additive Model (GAM)
* Neural Networks
* Model Assessment: AUC, Symmetric, and Asymmetric Misclassification Rate
* Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included**


**Major Findings and Conclusion**

![image](https://github.com/saidatsanni/Machine-Learning-Algorithms-for-Binary-Outcome-Variable/assets/139437600/1287ab25-5c90-4910-89c2-09858bc04aa4)

**The data set is split into 70% Training data and 30% Testing data**

* The advanced tree models: Bagging, Random Forest, and Boosting have a similar in-sample performance as the logistic model. However, these advanced tree methods perform slightly better than the classification tree, GAM, and neural network for in-sample prediction performance. Though, this may be due to variability given that the asymmetric misclassification rates for in-sample are close.
* The Logistic Regression Model and GAM have similar out-of-sample asymmetric misclassification. However, they perform better than the classification tree and the advanced tree models for out-of-sample prediction.
* Overall, the neural network has the best out-of-sample performance compared to all the models considered. In conclusion, the neural network does a much better job at predicting.


**Codes**

The codes can be found here: [Codes](https://github.com/saidatsanni/Machine-Learning-Algorithms-for-Binary-Outcome-Variable/blob/70dae7183498538339b5226bc7ade664231d3845/Main/Machine%20Learning%20on%20Binary_German%20Credit%20Data.R)
