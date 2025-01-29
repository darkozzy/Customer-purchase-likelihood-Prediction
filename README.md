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


# Model Performance Summary for Tripadvisor Review Dataset

## 1. Which Model Performed Best?

The **Naive Bayes** model achieved the following performance metrics:

- **Accuracy:** 81.82%  
- **Precision:** 73.29%  
- **Recall:** 81.82%  
- **F1 Score:** 76.45%  

While the overall accuracy and recall are strong, the model struggled with the **Neutral** class, as indicated by its classification report:

- **Positive Sentiment:** F1 Score = **0.90** → Excellent performance in identifying positive reviews.  
- **Negative Sentiment:** F1 Score = **0.67** → Moderate performance.  
- **Neutral Sentiment:** F1 Score = **0.00** → Significant challenges in accurately classifying neutral reviews.  

## 2. How Well Does It Meet the Business Objectives?

The business objective is to **classify customer reviews** as **Positive, Neutral, or Negative** using the `tripadvisor_review` dataset. Success is defined by achieving high performance on **Accuracy, Precision, Recall, and F1 Score**.

- The **Naive Bayes** model effectively identifies **Positive and Negative** sentiments, with high recall (**99% for Positive** and **55% for Negative**), ensuring that most positive and negative reviews are correctly classified.  
- However, the model **fails to perform well on Neutral sentiment classification** (**F1 Score = 0.00**), which may impact its ability to provide nuanced insights into customer feedback.  
- The overall **accuracy of 81.82%** demonstrates reliability for **binary classification (Positive vs. Negative)** but requires improvement for multi-class scenarios.  

## Key Insights

- **High recall for Positive sentiment** suggests that nearly all positive reviews are captured by the model.  
- **Misclassification of Neutral reviews** indicates that additional **feature engineering or an alternative algorithm** may be needed to improve performance in this category.  

## Recommendations for Improvement

- Implement **advanced text preprocessing** techniques (e.g., stemming, lemmatization) to enhance feature representation.  
- Experiment with other **algorithms** like **Support Vector Machines (SVM)** or **Gradient Boosting** to improve **Neutral** class detection.  
- Perform **hyperparameter tuning** on the **Naive Bayes** model to optimize its performance further.  
- Utilize **word embeddings** or **transformer-based models (e.g., BERT)** for better contextual understanding of text data.  

The **Naive Bayes** model provides a **strong baseline** but requires **further refinement** to fully meet the business objectives of accurately classifying all three sentiment categories.  

