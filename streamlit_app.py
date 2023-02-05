import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import numpy as np
from xgboost import XGBClassifier,Booster,DMatrix
import xgboost as xgb

class TitanicSurvival:
    def __init__(self, df):
        self.df = df
        self.columns_to_scale = ['Age', 'Fare']
        self.scaler = MinMaxScaler()
    
    def scale_columns(self):
        for column in self.columns_to_scale:
            if column in self.df.columns:
                self.df[column] = self.scaler.fit_transform(self.df[[column]])
        return self.df

    def label_encoder(self):
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column].astype(str))
        return self.df

    def reshaped_array(self):
        input_data_array = np.array(self.df)
        reshaped_array = input_data_array.reshape(1,-1)
        return reshaped_array
    
    def predict(self):
        self.scale_columns()
        self.label_encoder()
        clean_data = self.reshaped_array()


        try:
            # Load the model into the Booster object
            xgbModel = Booster()
            xgbModel.load_model('/Users/jamal/Desktop/streamlit/model.txt')

        except xgb.core.XGBoostError as e:
            # If there is an error, print the error message
            print(f"Error loading the model: {e}")

        try:
            # Predict using the loaded model and the data
            prediction = xgbModel.predict(DMatrix(clean_data))
            print('Prediction in the class',prediction)
            threshold = 0.5
            if prediction[0] >= threshold:
                return 'Survived'
            else:
                return 'Not Survived'

        except xgb.core.XGBoostError as e:
            # If there is an error, print the error message
            print(f"Error making prediction: {e}")
        
# ui components
# UI code for Streamlit
st.title("Titanic Survival Prediction :ship:")

# input fields
name = st.text_input('What is your name?:')
passenger_id = st.text_input('Input Passenger ID','1234')
passenger_class = st.select_slider("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age",0,100)
siblings_spouses = st.slider("Siblings/Spouses Aboard",0,8)#0-8
parents_children = st.slider("Parents/Children Aboard",0,3)#0-3
fare = st.number_input("Fare",value=0)
embarked = st.selectbox("Embarked", ["Southampton","Cherbourg", "Queenstown", ])
st.write('All of your infromation')
st.write('Name:',name)
st.write('Passenger ID::',passenger_id)
st.write('Passenger Class: ',passenger_class)
st.write('Gender: ',sex)
st.write('Age:',age)
st.write('Siblings/Spouses Aboard:',siblings_spouses)
st.write('Parent/Children Aboard:',parents_children)
st.write('Fare:',fare)
st.write('Embarked:',embarked)

user_data = {
    "Name":name,
    "Passenger ID":passenger_id,
    "Passenger class": passenger_class,
    "Gender": sex,
    "Age": age,
    "Siblings/Spouses Aboard": siblings_spouses,
    "Parent/Children Aboard": parents_children,
    "Fare": fare,
    "Embarked": embarked

}
display_dataframe = pd.DataFrame([user_data])
st.dataframe(display_dataframe,width=72,height=20,use_container_width=True)

# mapping values from input fields to a dictionary
data = {
    "Pclass": passenger_class,
    "Sex": sex,
    "Age": age,
    "SibSp": siblings_spouses,
    "Parch": parents_children,
    "Fare": fare,
    "Embarked": embarked
}

# preprocess the data and make a prediction
if st.button("Predict"):
    data = pd.DataFrame([data])
    print('Data from the user ',data)
    ts = TitanicSurvival(data)
    print('Class value outside the class',ts)
    prediction = ts.predict()
    print('Values received by streamlit',prediction)

    st.success(f'The Model prediction is successfull!', icon="✅")
    st.write('Results',prediction)
else:
    st.write('Please fill all the fields')

