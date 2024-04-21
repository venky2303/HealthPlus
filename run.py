import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

model = load('rfjoblib1.pkl')
hand_icon = "👋"

def load_data():
    df=pd.read_csv("final_datasetnewww.csv")
    return df
df=load_data()
def main():
    st.sidebar.title(f'Welcome Guys{hand_icon}')

    Age = st.slider("Age", min_value = 0, max_value = 100, value = 50, step = 1)
    Gender_selection = ['Male', 'Female']
    Gender_index = st.selectbox("Gender:", options=Gender_selection)
    Gender = 1 if Gender_index == 'Male' else 0
    
    Blood_Group_selection = ['O+', 'B-', 'O-', 'AB+', 'A+', 'A-', 'AB-', 'B+']  
    Blood_Group_index = st.selectbox("Blood Type:", options=Blood_Group_selection)
    Blood_Group_map = {'O+': 0,'B-': 1, 'O-': 2, 'AB+': 3, 'A+': 4, 'A-': 5, 'AB-': 6,'B+':7}
    Blood_Group = Blood_Group_map[Blood_Group_index]
    
    Medical_Condition_selection = ['Asthma', 'Obesity', 'Arthritis', 'Hypertension', 'Diabetes','Cancer']  
    Medical_Condition_index = st.selectbox("Medical Condition:", options=Medical_Condition_selection)
    Medical_Condition_map = {'Asthma': 0,'Obesity': 1, 'Arthritis': 2, 'Hypertension': 3, 'Diabetes': 4, 'Cancer': 5}  
    Medical_Condition = Medical_Condition_map[Medical_Condition_index]
    
    Medication_selection = ['Lipitor', 'Penicillin', 'Paracetamol', 'Aspirin', 'Ibuprofen']  
    Medication_index = st.selectbox("Medication:", options=Medication_selection)
    Medication_map = {'Lipitor': 0,'Penicillin': 1, 'Paracetamol': 2, 'Aspirin': 3, 'Ibuprofen': 4}  
    Medication = Medication_map[Medication_index]
    
    
    result=""
    if st.button("Predict"):
        result = predict_input(Age, Gender, Blood_Group, Medical_Condition, Medication)
    st.success("Result is {}".format(result))



    

with st.container():   
    icon_column, title_column = st.columns([1, 3])  
    with icon_column:    
        image_url = "stethoscope.jpg"
        st.image(image_url, use_column_width=False, width=150)
    with title_column:
        st.title('Healthcare Optimization Prediction')

st.write("---")
st.subheader("You can view the data below:")
parameters = ["Age Group", "Blood Type", "Medication", "Medical Condition", "Gender"]
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    selected_age_group = st.selectbox(" Age Group", df['Age'].unique())

with col2:
    selected_blood_type = st.selectbox("Blood Type", df['Blood Type'].unique())

with col3:
    selected_medication = st.selectbox(" Medication", df['Medication'].unique())

with col4:
    selected_medical_condition = st.selectbox(" Medical Condition", df['Medical Condition'].unique())

with col5:
    selected_gender = st.selectbox(" Gender", df['Gender'].unique())

filtered_df = df[
    (df['Age'] == selected_age_group) &
    (df['Blood Type'] == selected_blood_type) &
    (df['Medication'] == selected_medication) &
    (df['Medical Condition'] == selected_medical_condition) &
    (df['Gender'] == selected_gender)
]
st.write(filtered_df)
fig, ax = plt.subplots()
filtered_df['Gender'].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
ax.set_title('Gender Distribution')
ax.set_ylabel('')
st.pyplot(fig)

def predict_input(Age, Gender, Blood_Group, Medical_Condition, Medication):
    print("Input data:", Age, Gender, Blood_Group, Medical_Condition, Medication)  
    prediction = model.predict([[Age, Gender, Blood_Group, Medical_Condition, Medication]])
    label_map = {0: 'Normal', 1: 'Abnormal'}
    predicted_label = label_map[prediction[0]]
    return predicted_label



if __name__=='__main__':
    main()
