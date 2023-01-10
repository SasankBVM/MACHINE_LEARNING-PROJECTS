import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as gb
import pickle
st.title("DIABETES PREDICTION")
st.set_option('deprecation.showPyplotGlobalUse', False)
data = pd.read_csv("diabetes.csv")
data.drop('Pregnancies',axis=1,inplace=True)
nav = st.sidebar.radio("NAVIGATION", ["HOME", "PREDICT"])
if nav == "HOME":
    st.image("diab.jpg", 50, 50, True)
    st.markdown(""" ### INFORMATION ABOUT THE DATASET""")
    if st.checkbox("DETAILED DESCRIPTION"):
        st.markdown("""
        ### DATA EXPLANATION
        #### 1) Pregnancies- Denotes the number of pregnancies the patient had.
        #### 2) Glucose- Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
        #### 3) BloodPressure-Diastolic blood pressure (mm Hg)
        #### 4) SkinThickness- Triceps skin fold thickness (mm)
        #### 5) Insulin- 2-Hour serum insulin (mu U/ml)
        #### 6) BMI -Body mass index (weight in kg/(height in m)^2)
        #### 7) DiabetesPedigreeFunction- Values of study of whether the subject's family members had Diabetes.
        #### 8) Age- Age(years)
        #### 9) Outcome -->1(If had Diabetes),  -->0(If don't)
        """)
    st.markdown("""### DATASET: DIABETES.CSV""")
    st.dataframe(data)
    if st.checkbox("SHOW TABLE"):
        st.table(data)
    st.markdown(""" ## EXPLORATORY DATA ANALYSIS""")
    if st.checkbox("BLOOD PRESSURE VS INSULIN LEVELS"):
        plt.figure(figsize=(10, 5))
        plt.scatter(data['BloodPressure'], data['Insulin'])
        plt.xlabel("BLOOD PRESSURE")
        plt.ylabel("INSULIN LEVEL")
        st.pyplot()

        plt.figure(figsize=(10, 5))
        plt.bar(data['BloodPressure'], data['Insulin'])
        plt.xlabel("BLOOD PRESSURE")
        plt.ylabel("INSULIN LEVEL")
        st.pyplot()
    if st.checkbox("GLUCOSE VS INSULIN LEVELS"):
        plt.figure(figsize=(10, 5))
        plt.scatter(data['Glucose'], data['Insulin'])
        plt.xlabel("GLUCOSE")
        plt.ylabel("INSULIN LEVEL")
        st.pyplot()

        plt.figure(figsize=(10, 5))
        plt.bar(data['Glucose'], data['Insulin'])
        plt.xlabel("GLUCOSE")
        plt.ylabel("INSULIN LEVEL")
        st.pyplot()
        # st.write("INTERACTIVE PLOT")
        # layout=gb.Layout(
        #     xaxis=dict(range=[0,2000]),
        #     yaxis=dict(range=[0,2000])
        # )
        # fig=gb.Figure(data=gb.Bar(x=data['Pregnancies'],y=data['Insulin'],mode='marker'),layout=layout)
        # st.plotly_chart(fig)
elif nav == "PREDICT":
    model=pickle.load(open('finalized.sav', 'rb'))
    st.markdown("""EXAMPLE DATA [138,62,35,0,33.6,0.127,47] HAD DIABETES """)
    st.markdown("""EXAMPLE DATA [84,82,31,125,38.2,0.233,23] DOESN'T HAVE DIABETES """)
    gluco=st.number_input("ENTER YOUR GLOCOSE LEVEL",0,200,value=138)
    bp=st.number_input("ENTER BLOOD PRESSURE",0,123,value=62)
    skt=st.number_input("ENTER YOUR GLOCOSE LEVEL",0,111,value=35)
    insulin=st.number_input("ENTER YOUR INSULIN LEVEL",0,745,value=0)
    BMI=st.number_input("ENTER YOUR BODY-MASS-INDEX",0.0,81.5,value=33.6)
    dpf=st.number_input("ENTER YOUR DPF",0.0,2.45,value=0.127)
    age=st.number_input("ENTER YOUR AGE",1,100,value=47)
    inputs=[gluco,bp,skt,insulin,BMI,dpf,age]
    inputs=np.array(inputs).reshape(1,-1)
    pred=model.predict(inputs)
    if st.button("PREDICT"):
        if pred==0:
            st.balloons()
            st.write("YOU DON'T HAVE DIABETES")
        elif pred==1:
            st.write("YOU HAVE DIABETES")