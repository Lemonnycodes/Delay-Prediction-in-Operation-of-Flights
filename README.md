# Flight Delay Prediction and Analysis

This repository contains code for predicting flight delays and analyzing the factors contributing to flight delays. The project is implemented in Python and utilizes libraries such as pandas, scikit-learn, matplotlib, and seaborn. 

## Project Overview
Flight delays can cause significant inconvenience and disruptions for travelers. This project aims to predict flight delays and analyze the patterns and factors that contribute to flight delays. By utilizing historical flight data and machine learning techniques, the project provides insights into flight delay prediction and helps identify potential areas for improvement in airline operations.

## Tech Stack
The project is implemented using the following technologies:
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
  ## Evaluation Results

The model evaluation results for the Bernoulli Naive Bayes Classifier with the best parameter value of alpha=1 are as follows:

- Testing Score: The model achieved a testing score of 0.8467, indicating an accuracy of approximately 84.67% on the test set.

- Precision and Recall: For the positive class, the precision is 0.74, indicating that 74% of the instances predicted as positive are actually positive. The recall (sensitivity) is 0.45, indicating that the model correctly predicted 45% of the actual positive instances.

- F1-Score: The F1-score, which is a balanced measure of precision and recall, for the positive class is 0.56.

- Support: The support refers to the number of instances in each class in the test set. In this case, there are 1,206,371 instances in the negative class (0.0) and 333,548 instances in the positive class (1.0).

- Confusion Matrix: The confusion matrix provides detailed information about the model's predictions. For the Bernoulli Naive Bayes Classifier, the confusion matrix is as follows:

    |            | Predicted Negative | Predicted Positive |
    |------------|--------------------|--------------------|
    | Negative   | 1,153,728          | 52,643             |
    | Positive   | 183,423            | 150,125            |

- Normalized Confusion Matrix: The normalized confusion matrix presents the relative proportions of the predicted classes. The normalized confusion matrix for the Bernoulli Naive Bayes Classifier is as follows:

    |            | Predicted Negative | Predicted Positive |
    |------------|--------------------|--------------------|
    | Negative   | 0.9564             | 0.0436             |
    | Positive   | 0.5499             | 0.4501             |

These evaluation results provide insights into the performance of the Bernoulli Naive Bayes Classifier in predicting flight delays. The metrics highlight the precision, recall, F1-score, and accuracy for both the positive and negative classes. The confusion matrix and normalized confusion matrix illustrate the distribution of predicted classes and the model's accuracy in each class.


### Results
The evaluation results for the Bernoulli Naive Bayes Classifier model with the best parameter value of alpha=1 are as follows:

- Testing Score: The model achieved a testing score of 0.8467, indicating an accuracy of approximately 84.67% on the test set.

- Precision and Recall: For the positive class, the precision is 0.74, indicating that 74% of the instances predicted as positive are actually positive. The recall (sensitivity) is 0.45, indicating that the model correctly predicted 45% of the actual positive instances.

The project aims to predict flight delays and provide insights into the factors contributing to flight delays. The specific results achieved, such as prediction accuracy, model evaluation metrics, and analysis findings, can be described based on the execution of the project code and analysis.




