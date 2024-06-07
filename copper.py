import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn .tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from joblib import load


st.set_page_config(layout="wide")

st.title("Industrial Copper Modeling Application")



product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
        '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
        '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
        '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
        '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
status_menu = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised', 'Offered', 'Offerable']


with st.form('form1'):
    col1,col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country",country_options,index=None,placeholder="Select")
        status = st.selectbox("Status",status_menu,index=None,placeholder="Select")
        item_type = st.selectbox("Item type",['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'],index=None,placeholder="Select")
        application = st.selectbox("Application",application_options,index=None,placeholder="Select")            
        product_ref = st.selectbox("Product Reference",product,index=None,placeholder="Select")
        

    with col2:
        st.write( f'<h5 style="color:red;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
        quantity = st.text_input("Enter Quantities Ton (Min:611728 & Max:1722207579)")
        customer = st.text_input("Enter Customer Id  (Min:12458, Max:30408185)")
        thickness = st.text_input("Enter Thickness  (Min:0.18 & Max:400)")
        width = st.text_input("Enter Width (Min:1,  Max:2990)")            
        predict_button = st.form_submit_button("Predict Selling Price")
        st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)
# column order ["quantity tons","customer","country","application","thickness","width","product_ref",'status','item type']      
if predict_button:
    model = load('copperRegressorModel.joblib')
    ohe = load('statusOHE.joblib')
    ohe1 = load('itemtypeOHE.joblib')
    scale = load('stdscaler.joblib')
    sample = np.array([[np.log(float(quantity)),float(customer),country,application,np.log(float(thickness)),float(width),int(product_ref),status,item_type]])
    sohe = ohe.transform(sample[:,[7]]).toarray()
    iohe = ohe1.transform(sample[:,[8]]).toarray()
    sample = np.concatenate((sample[:,[0,1,2,3,4,5,6]],sohe,iohe),axis=1)        
    sample = scale.transform(sample)
    prediction = model.predict(sample)[0]
    price = f"$ {np.exp(prediction)}"
    st.write('## :green[Predicted Selling Price:] ',price )




        