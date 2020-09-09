# Predicting Diagnosis of Urinary Bladder Inflammation & Nephritis:
## Sklearn Implementation using Pipelines, Grid Search, and Ensemble Learning

## Overall Summary¶
We used archived data downloaded from the UCI machine learning repository containing features and outcomes for two diseases of the urinary bladder: acute inflammation acute nephritis. Data was preprocessed by label encoding all binary features and then standardizing all input features.

Due to the nature of the data (1 continuous feature and 5 binary features) we were limited with traditional parametric methods. Therefore, we choose models robust to non-linear features: k-nearest neighbors, decision tree, random forest, and SGD (SVM implementation). To find the best parameters, a grid search was completed for each algorithm. With the best model chosen, the following were calculated: accuracy score and running time. This process was completed for both target variables.

The following are the best parameters for each algorithm:

Target: Inflammation

KNN:{'metric': 'chebyshev', 'n_neighbors': 1}

Decision Tree: {'criterion': 'gini', 'max_depth': 3}

Random Forest: {'criterion': 'gini', 'n_estimators': 50}

SGD(SVM): {'alpha': 0.0001, 'loss': 'hinge', 'penalty': 'l2'}

Target: Nephritis

KNN: {'metric': 'chebyshev', 'n_neighbors': 1}

Decision Tree: {'criterion': 'gini', 'max_depth': 2}

Random Forest: {'criterion': 'gini', 'n_estimators': 50}

SGD(SVM: {'alpha': 0.001, 'loss': 'hinge', 'penalty': 'l2'}


Overall, all models performed well but KNN seems to be the best choice when considering run-time. If computational cost is not an issue, random forest would also be a great choice. Thankfully, we were able to model the data how we had hoped. Other graduate students had published previous work on this dataset and we were able to read their publication as guidance and explore areas they had not.

Here is how our work differs from this publication:

Prior work had combined the two outcome variables; we analyzed them separately for interpretability.

Prior work did not use nested cross-validation (they use hold-out cross-validation)

Either prior work did not use grid search to find best parameters (as we did) or simply did not mention it in their paper.

While prior work did use variations of KNN, our grid search's best result produced a different parameters(metric=chebyshev) than any of their variations(Euclidean, Manhattan, Cosine, Correlation).

While prior work used variations of SVM, SGD (stochastic gradient descent) was not implemented in any of those variations as we did.

Finally, what did we learn? First, we learned that the best features for identifying the presence of inflammation ('lumbar_pain', 'pushing', 'nausea', 'burning') were not exactly the same as those for identifying nephritis ('burning', 'nausea','temp', ''micturition'). This result supports the notion that these two target variables should be analyzed separately, depending on the application, of course.

What was most surprising was the best parameter finding for KNN. The prior work had not even tried the chebyshev metric, and yet we found this metric to be the best. While adding the run-times was not required, it was a great learning tool as we were able to see the computational cost of random forests as compared to other algorithms. Overall, this project was a great tool for learning how to implement grid search and cross validation, as well as an opportunity to expand among previous machine learning research published on this data.
