# Machine Learning Algorithms for Binary Outcome Variable Prediction

**Goal**

The goal of this study is to conduct a comprehensive predictive analysis of credit default probability and identify the model that provides the best prediction accuracy. Machine Learning methods such as Logistic Regression Models, Variable Selection, Classification Trees, Bagging, Random Forest, Boosted Regression Trees, Generalized Additive Models, and Neural Networks are considered. The model performances are assessed using the Area Under the Curve and Misclassification Rate.


**Data**

The German credit score dataset consists of 7000 observations and 21 different quantitative variables. 

**Statistical Techniques**

* Generalized Linear Model (Logistic Regression Analysis)
* Variable and Model Selection 
* Classification Trees
* Bagging
* Random Forest
* Boosted Regression Trees
* Generalized Additive Model (GAM)
* Neural Networks
* Model Assessment: AUC curve, Symmetric, and Asymmetric Misclassification Rate
* Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included**


**Major Findings and Conclusion**

[Model Comparisonhttps://github.com/saidatsanni/Machine-Learning-Algorithms-for-Binary-Outcome-Variable/blob/a9e9cadfe76988ec34709bc649e55b7a78738e09/Model%20Comparison_Credit.jpg


**The data set is split into 70% Training data and 30% Testing data**
**AMR represents Asymmetric Misclassification Rate (5:1 asymmetric cost)**

* The advanced tree models: Bagging, Random Forest, and Boosting have a similar in-sample performance as the logistic model. However, these advanced tree methods perform slightly better than the classification tree, GAM, and neural network for in-sample prediction performance. Though, this may be due to variability given that the asymmetric misclassification rates for in-sample are close.
* The Logistic Regression Model and GAM have similar out-of-sample asymmetric misclassification. However, they perform better than the classification tree and the advanced tree models for out-of-sample prediction.
* Overall, the neural network has the best out-of-sample performance compared to all the models considered. In conclusion, the neural network does a much better job at predicting.


**Codes**

The codes can be found here: [Codes](https://github.com/saidatsanni/Machine-Learning-Models-on-Boston-Housing-Data/blob/0b304a99c9f387ad17593c0754721ffb939d45b0/Main/Machine%20Learning%20on%20Boston%20Housing%20Data.R)
