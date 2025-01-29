# Customer-purchase-likelihood-Prediction

# Business Understanding

## Objectives

- **Predict customer purchase likelihood** using the `online_shoppers_intention` dataset.  
- **Classify customer reviews** as positive or negative using the `tripadvisor_review` dataset.  
- **Predict customer churn** using the `Telco_Customer_Churn_Dataset`.  

## Success Metrics

- **Accuracy, Precision, Recall, and F1 Score** for all models.  
- **ROC AUC** for binary classification tasks.  

## Research Questions

- **What factors influence online shopping behavior?**  
- **Can text-based reviews be effectively classified using Naive Bayes?**  
- **Which features are most important for predicting customer churn?**  



# Model Performance Summary

## 1. Which Model Performed Best?

The **Gradient Boosting** model achieved the best overall performance across multiple metrics:

- **Accuracy:** 90.06%  
- **Precision:** 72.17%  
- **Recall:** 58.38%  
- **F1 Score:** 64.54%  
- **ROC AUC:** 92.59%  

The **Gradient Boosting** model outperformed other models in terms of **ROC AUC**, which is a critical metric for binary classification tasks like predicting customer purchase likelihood.

## 2. How Well Does It Meet the Business Objectives?

The business objective is to **predict customer purchase likelihood** using the `online_shoppers_intention` dataset. Success is defined by achieving high performance on key metrics such as **Accuracy, Precision, Recall, F1 Score, and ROC AUC**.

- The **Gradient Boosting** model aligns well with the business objective as it provides a balance between **Precision (72.17%)** and **Recall (58.38%)**, ensuring fewer false positives and false negatives.
- Its high **ROC AUC score (92.59%)** indicates the model's strong ability to distinguish between customers likely to make a purchase and those who are not.
- While **Recall** could be improved further to capture more true positives, the model's overall performance is **robust and suitable for deployment** in predicting customer purchase likelihood.

## Selected Final Model

The **Gradient Boosting** model has been selected as the **final model** based on its superior performance across all evaluation metrics.  

The trained model has been saved as **`final_model.pkl`** for deployment purposes.
