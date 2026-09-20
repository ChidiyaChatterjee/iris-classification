# Iris Classification

This project is a simple machine learning classification task using the classic Iris dataset from scikit-learn. The notebook loads the dataset, explores it, compares several classification models, selects the best-performing model, evaluates it, and saves the trained model for later use.

## Project Goal
The goal is to classify iris flowers into one of three species:
- Setosa
- Versicolor
- Virginica

The model uses flower measurements such as:
- sepal length
- sepal width
- petal length
- petal width

## Dataset
The dataset used is the built-in scikit-learn Iris dataset.

### Dataset Details
- Total samples: 150
- Features: 4 numeric attributes
- Target classes: 3 species
- Source: scikit-learn

## What is in the Notebook
The notebook contains a complete beginner-friendly machine learning workflow:

1. Import required libraries
   - pandas
   - scikit-learn
   - seaborn
   - matplotlib
   - joblib

2. Load the Iris dataset

3. Create feature and target variables
 
4. Convert data into a DataFrame
   - Add feature names as columns
   - Add target labels
   - Add species names using the target mapping

5. Explore the dataset
   - check shape
   - check missing values
   - check class distribution
   - view the first rows
   - view summary statistics

6. Visualize the data
   - create a pair plot using seaborn
   - color by species to see class separability

7. Split the data
   - use train and test sets
   - maintain class balance using stratification

8. Compare multiple models
   - Logistic Regression
   - SVM
   - KNN
   - Decision Tree
   - Random Forest

9. Evaluate model performance
   - accuracy score
   - classification report
   - confusion matrix

10. Perform cross-validation
   - validate the chosen model using multiple folds

11. Train the final model
   - K-Nearest Neighbors (KNN) was selected as the best-performing model in this notebook

12. Evaluate the final trained model
   - print classification report
   - print confusion matrix
   - print final accuracy

13. Visualize the confusion matrix
   - use seaborn heatmap

14. Save the trained model

## Best Model
In this project, the K-Nearest Neighbors (KNN) classifier achieved the best accuracy on the test set, so it was selected as the final model for model evaluation and saving.

## Libraries Used
- pandas
- scikit-learn
- seaborn
- matplotlib
- joblib

## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open the notebook in Jupyter Notebook or VS Code.
2. Run all cells in order.
3. The notebook will:
   - load data
   - visualize patterns
   - compare models
   - train the best model
   - print evaluation metrics
   - save the model

## Output
The notebook generates:
- model comparison results
- cross-validation scores
- classification report
- confusion matrix
- heatmap visualization
- saved model file: `iris_knn_model.pkl`

## Notes
- The dataset is small and easy to classify, so model performance is usually very high.
- KNN performs especially well on this dataset because the classes are well separated.
- This project is a good beginner example of a complete machine learning workflow.

## Files in the Project
- `iris_classifier.ipynb` — main notebook
- `requirements.txt` — Python dependencies
- `iris_knn_model.pkl` — saved trained model 

## License
This project is intended for learning and educational purposes.