This project demonstrates how to apply linear regression to predict housing prices in California based on various socioeconomic and geographical features. It includes data preprocessing, model training, outlier removal, evaluation, and visualization.

Project Overview
The notebook includes the following steps:
Load the California Housing dataset from sklearn.datasets
Split the data into training and testing sets
Fit a linear regression model
Analyze residuals to detect and remove outliers
Evaluate the model performance using R² score and Mean Squared Error
Visualize the results using standard diagnostic plots

Technologies Used
Python
pandas and numpy for data manipulation
scikit-learn for model training and evaluation
matplotlib and seaborn for data visualization

Step-by-Step Breakdown
Data Loading
The dataset is loaded using fetch_california_housing() from scikit-learn.
Train/Test Split
The data is split 80/20 to train and test the model effectively.

Model Training
A linear regression model is trained using the default parameters.

Residual Calculation
Residuals (actual - predicted values) are calculated to assess prediction errors.

Outlier Detection and Removal
Outliers in residuals are removed using the Interquartile Range (IQR) method to ensure better model performance.

Evaluation Metrics
The model is evaluated using:
R² score — explains variance
Mean Squared Error — average squared difference between actual and predicted values

Visualizations

Residual Distribution: Histogram with KDE to check normality of residuals
Actual vs Predicted: Scatter plot to assess prediction accuracy

How to Use
Run the notebook cell by cell. Make sure you have all required libraries installed. You can use pip:

 -pip install pandas numpy matplotlib seaborn scikit-learn

Potential Improvements
Add regularization techniques like Ridge or Lasso regression

Experiment with polynomial features
Use cross-validation to validate the model
Add feature scaling or feature selection

File Structure
california-housing-regression/
├── california_housing_linearregression.py  # Jupyter notebook with full analysis
└── README.md                # Project description and instructions

