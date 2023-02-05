# Titanic Survival Prediction_Streamlit App
This app is an implementation of a machine learning model that predicts the survival of a passenger on the Titanic ship. The model is built using XGBoost, an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. The UI of the app is built using Streamlit, a Python library for building ML and data science apps.

 # Prerequisites
The following packages are required to run the code:

streamlit
pandas
joblib
scikit-learn
numpy
xgboost
You can install these packages by running pip install <package-name>.

 # Usage
The app allows the user to enter the following information:

Name
Passenger ID
Passenger Class (1, 2 or 3)
Gender
Age
Siblings/Spouses Aboard
Parents/Children Aboard
Fare
Embarked (Southampton, Cherbourg or Queenstown)
 
Based on the entered information, the app makes a prediction of whether the passenger would have survived or not. To make the prediction, the input values are preprocessed to prepare them for use with the machine learning model. The preprocessing steps include:

Scaling of Age and Fare columns using MinMaxScaler.
Encoding of categorical columns using LabelEncoder.
After preprocessing, the data is passed to the XGBoost model, which returns a prediction of either 'Survived' or 'Not Survived'.

 # File Structure
The code contains the following files:

streamlit_app2.py: This is the main code file that contains the class TitanicSurvival that performs all the preprocessing and prediction tasks.
# Running the app
To run the app, navigate to the folder where the code is located and run the following command in your terminal:

Copy code
streamlit run streamlit_app2.py
The app will then open in your default browser. Fill in the input fields and click the 'Predict' button to get the survival prediction.

# Note
The model file model.txt is required to run the app. Make sure to provide the correct file path in the code.
