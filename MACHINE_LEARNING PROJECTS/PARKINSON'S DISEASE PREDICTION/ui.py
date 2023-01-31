import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from joblib import Parallel, delayed
import joblib
st.title("PARKINSON'S DISEASE PREDICTION")
st.set_option('deprecation.showPyplotGlobalUse', False)
data = pd.read_csv("./Parkinsson disease.csv")
data.drop('name',axis=1,inplace=True)
nav = st.sidebar.radio("NAVIGATION", ["HOME", "PREDICT","PERFORMANCE METRICS"])
model=joblib.load(open('svm_model.pkl', 'rb'))
if nav == "HOME":
    st.image("park.png", 50, 50, True)
    st.markdown(""" ### INFORMATION ABOUT THE DATASET""")
    if st.checkbox("DETAILED DESCRIPTION"):
        st.markdown("""
        ### MDVP:F0 (Hz)	Average vocal fundamental frequency
        ### MDVP:Fhi (Hz)	Maximum vocal fundamental frequency
        ### MDVP:Flo (Hz)	Minimum vocal fundamental frequency
        ### MDVP:Jitter(%)	MDVP jitter in percentage
        ### MDVP:Jitter(Abs)	MDVP absolute jitter in ms
        ### MDVP:RAP	MDVP relative amplitude perturbation
        ### MDVP:PPQ	MDVP five-point period perturbation quotient
        ### Jitter:DDP	Average absolute difference of differences between jitter cycles
        ### MDVP:Shimmer	MDVP local shimmer
        ### MDVP:Shimmer(dB)	MDVP local shimmer in dB
        ### Shimmer:APQ3	Three-point amplitude perturbation quotient
        ### Shimmer:APQ5	Five-point amplitude perturbation quotient
        ### MDVP:APQ11	MDVP 11-point amplitude perturbation quotient
        ### Shimmer:DDA	Average absolute differences between the amplitudes of consecutive periods
        ### NHR	Noise-to-harmonics ratio
        ### HNR	Harmonics-to-noise ratio
        ### RPDE	Recurrence period density entropy measure
        ### D2	Correlation dimension
        ### DFA	Signal fractal scaling exponent of detrended fluctuation analysis
        ### Spread1	Two nonlinear measures of fundamental
        ### Spread2	Frequency variation
        ### PPE	Pitch period entropy
        """)
    st.markdown("""### DATASET: PARKINSON DISEASE.CSV""")
    st.dataframe(data)
    if st.checkbox("SHOW TABLE"):
        st.table(data)
    st.markdown(""" ## EXPLORATORY DATA ANALYSIS""")
    if st.checkbox("FEATURES WITH MAXIMUM CORRELATION WITH TARGET CLASS"):

        fig_=plt.figure(figsize=(10, 5))
        sns.countplot(x=data['status'])
        plt.xlabel("NUMBER OF DISTINCT TARGET CLASS LABELS")
        plt.ylabel("COUNT")
        st.pyplot(fig_)

        fig=plt.figure(figsize=(10, 5))
        plt.title("PPE VS SPREAD-1")
        sns.regplot(x=data['PPE'], y=data['spread1'])
        plt.xlabel("PITCH PERIOD ENTROPY")
        plt.ylabel("SPREAD-1")
        st.pyplot(fig=fig)

        fig_=plt.figure(figsize=(10, 5))
        plt.title("MDVP:JITTER VS MDVP:RAP")
        sns.regplot(x=data['MDVP:Jitter(Abs)'],y=data['MDVP:RAP'])
        plt.xlabel("'MDVP:Jitter(Abs)")
        plt.ylabel("MDVP:RAP")
        st.pyplot(fig_)
    elif st.checkbox("CLICK HERE TO SEE THE RELATION AMONG ALL COLUMNS"):
        x_vals=st.selectbox("SELECT X-AXIS COLUMNS",options=data.columns)
        y_vals=st.selectbox("SELECT Y-AXIS COLUMNS",options=data.columns)
        plot=px.scatter(data,x=x_vals,y=y_vals)
        plot.update_layout(
        font=dict(
        family="Calibri",
        size=20,  # Set the font size here
        color="green"
    )
)
        if st.button("PLOT"):
            st.plotly_chart(plot)

        # st.write("INTERACTIVE PLOT")
        # layout=gb.Layout(
        #     xaxis=dict(range=[0,2000]),
        #     yaxis=dict(range=[0,2000])
        # )
        # fig=gb.Figure(data=gb.Bar(x=data['Pregnancies'],y=data['Insulin'],mode='marker'),layout=layout)
        # st.plotly_chart(fig)
# 0.121009,1.737287,-0.860817,-0.944288,-0.757011,-0.823129,-0.758083,-0.971711,
# -0.924538,-0.993521,-0.888872,-0.876144,-0.993519,-0.589551,2.181916,-1.431704,
# 0.431668,-1.870140,-0.649697,0.137057,-1.616379

# -0.994279,-0.758967,-0.152844,0.764967,1.232424,0.611243,1.425912,0.612212,1.058449,0.885794
# ,1.036706,1.570486,0.576632,1.036405,-0.324961,-0.209630,-0.629027,1.974581,
# 1.087645,1.009449,-0.127555,1.333395
elif nav == "PREDICT":
    st.markdown("""EXAMPLE DATA [119.992,157.302,74.997,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.426,0.02182,0.03130,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654] HAD THE DISEASE """)
    st.markdown("""EXAMPLE DATA [197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569] DOESN'T HAVE DISEASE """)
    mdvp_1=st.selectbox("ENTER MDVP:F0 ",(119.992,1.116545,-0.994279))
    mdvp_2=st.selectbox("ENTER MDVP:Fhi(Hz)	",(157.302,0.121009,-0.758967))
    mdvp_3=st.selectbox("ENTER MDVP:Flo (Hz)",(74.997,1.737287,-0.152844))
    mdvp_4=st.selectbox("ENTER MDVP:Jitter(%)",(0.00784,-0.860817,0.764967))
    mdvp_5=st.selectbox("ENTER MDVP:Jitter(Abs)",(0.00007,-0.944288,1.232424))
    mdvp_6=st.selectbox("ENTER MDVP:RAP",(0.00370,-0.757011,0.611243))
    mdvp_7=st.selectbox("ENTER MDVP:PPQ",(0.00554,-0.823129,1.425912))
    jitter=st.selectbox("ENTER Jitter:DDP",(0.01109,-0.758083,0.612212))
    mdvp_8=st.selectbox("ENTER MDVP:Shimmer",(0.04374,-0.971711,1.058449))
    mdvp_9=st.selectbox("ENTER MDVP:Shimmer(dB)",(0.426,-0.924538,0.885794))
    shimmer_1=st.selectbox("ENTER Shimmer:APQ3",(0.02182,-0.993521,1.036706))
    shimmer_2=st.selectbox("ENTER Shimmer:APQ5",(0.03130,-0.888872,1.570486))
    mdvp_10=st.selectbox("ENTER MDVP:APQ11",(0.02971,-0.876144,0.576632))
    shimmer_3=st.selectbox("ENTER Shimmer",(0.06545,-0.993519,1.036405))
    nhr=st.selectbox("ENTER NHR",(0.02211,-0.589551,-0.324961))
    hnr=st.selectbox("ENTER HNR",(21.033,2.181916,-0.209630))
    rdpe=st.selectbox("ENTER RDPE",(0.414783,-1.431704,-0.629027))
    d2=st.selectbox("ENTER D2",(0.815285,0.431668,1.974581))
    dfa=st.selectbox("ENTER DFA",(-4.813031,-1.870140,1.087645))
    spread_1=st.selectbox("ENTER SPREAD_1",(0.266482,-0.649697,1.009449))
    spread_2=st.selectbox("ENTER SPREAD_2",(2.301442,0.137057,-0.127555))
    ppe=st.selectbox("ENTER PPE",(0.284654,-1.616379,1.333395))
    inputs=[mdvp_1,mdvp_2,mdvp_3,mdvp_4,mdvp_5,mdvp_6,mdvp_7,jitter,mdvp_8,mdvp_9,shimmer_1,shimmer_2,mdvp_10,shimmer_3,nhr,hnr,rdpe,d2,dfa,spread_1,spread_2,ppe]
    inputs=np.array(inputs).reshape(1,-1)
    pred=model.predict(inputs)
    if st.button("PREDICT"):
        if pred==0:
            st.balloons()
            st.write("YOU DON'T HAVE DISEASE")
        elif pred==1:
            st.write("YOU HAVE THE DISEASE")
    # if st.checkbox("SUPPORT VECTOR MACHINE"):
    #     st.markdown(model.score(inputs,data[[inputs[0]==data['MDVP:Fo(Hz)']]]['status'])) 
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)