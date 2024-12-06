# project1
Overview
The model is a Logistic Regression classifier trained to predict the training score based on several independent features related to employee performance and training history. The model is implemented using Python's scikit-learn library and is deployed through a Streamlit web application for user interaction.
Model Details
Model Type: Logistic Regression
Library Used: scikit-learn
Random State: 0 (for reproducibility)
Feature Descriptions
The following independent features are used as input for the model:
Age:
Type: Integer
Range: 20 to 60
Description: The age of the employee.
Length of Service:
Type: Integer
Range: 1 to 34 years
Description: The number of years the employee has been with the company.
Number of Trainings:
Type: Integer
Range: 1 to 10
Description: The total number of training sessions attended by the employee.
Previous Year Rating:
Type: Integer
Range: 1 to 5
Description: The performance rating of the employee from the previous year.
KPIs Met More Than 80%:
Type: Binary (0 or 1)
Description: Indicates whether the employee met more than 80% of their Key Performance Indicators (KPIs).
Awards Won:
Type: Binary (0 or 1)
Description: Indicates whether the employee has won any awards.
Region:
Type: Categorical (integer encoding)
Description: Represents the region where the employee is located.
Education Level:
Type: Categorical (encoded as integers)
Bachelors = 0
Masters = 1
PhD = 2
Description: The highest level of education attained by the employee.
Data Preprocessing
Before training, the data undergoes preprocessing, which includes:
Encoding categorical variables into numerical format.
Normalizing or scaling features if necessary (not explicitly mentioned in this documentation but often recommended).
Training Process
The model is trained using historical data that includes features listed above and their corresponding training scores. The training score is treated as the target variable, which the model learns to predict based on input features.
Usage Instructions
To use the Streamlit application:
Run the Application:
Open your terminal or command prompt and navigate to the directory containing app.py. Run:
bash
streamlit run app.py

Input Features:
On the prediction page, enter values for each independent feature using the provided input fields.
Make a Prediction:
Click on the "Predict Score" button to generate a prediction for the training score based on your inputs.
View Results:
The predicted score will be displayed below the input fields.
Example Input
Feature	Example Value
Age	30
Length of Service	5
Number of Trainings	2
Previous Year Rating	4
KPIs Met More Than 80%	1
Awards Won	0
Region	2
Education Level	Masters
Conclusion
This documentation provides an overview of how to use and understand the logistic regression model implemented in the Streamlit app for predicting training scores based on various employee-related features. Adjustments can be made to improve accuracy or incorporate additional features as needed based on further analysis and testing.
