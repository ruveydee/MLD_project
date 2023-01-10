import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image 

# here we define some of the front end elements of the web page like the font and background color,
# the padding and the text to be displayed


html_temp = """
	<div style ="background-color:#F1FA18; padding:13px">
<h1 style ="color:black; text-align:center; "> Auto Scout Car Price Prediction
    </h1>
	</div>
    
	"""
st.markdown(html_temp, unsafe_allow_html = True)
image = Image.open("auto_scout24.png")
st.image(image, use_column_width=True)

# Display Auto Scout Dataset
html_head = "[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=&lines=+Hi+!+,+Welcome+to+Auto+Scout24;Ready+to+the+Car+Price+Prediction!)](https://git.io/typing-svg)"
st.markdown(html_head,unsafe_allow_html = True )

st.info("For this project we are using a car dataset, where we want to predict the selling price of car based on its certain features. Since we need to find the real value, with real calculation, therefore this problem is regression problem. We will be using linear regression to solve this problem.")

#load the dataset
df = pd.read_csv('final_scout_not_dummy2.csv')
st.write(df.head())
st.write('_Shape of the data_:',df.shape)
st.write(df.describe())

# Loading the models to make predictions
linear_model = pickle.load(open("final_linear_pipe_model", "rb"))


# User input variables that will be used on predictions
st.sidebar.title("_Please Enter the Features and Model Name to predict the price of car_")
make_model = st.sidebar.selectbox('Select the model of the car',('Audi A3','Audi A1','Opel Insignia','Opel Astra','Opel Corsa','Renault Clio','Renault Espace','Renault Duster'))
hp_kw= st.sidebar.slider("Horse Power(kW)",min_value=30.000,max_value=300.000, step=1.000)
km= st.sidebar.slider("km",min_value=0.000,max_value=500.000, step=1.000)
age= st.sidebar.slider("Age",min_value=0,max_value=15,step=1)
gearing_type= st.sidebar.selectbox('Select the Gearing Type',("Manual","Automotic","Semi-automatic"))
gears= st.sidebar.slider("Gears",min_value=3,max_value=10,step=1)
type= st.sidebar.selectbox("Type of the car",("Used","New","Pre-registered","Employee's car","Demonstration"))
safety_securty= st.sidebar.selectbox('Select package',("Safety Premium Package","Safety Premimum Plus Package","Safety Standart Package"))


my_dict = {
    "make_model": make_model,
    "hp_kW": hp_kw,
    "km": km,
    "age": age,
    "Gearing_Type": gearing_type,
    "Gears": gears,
    "Type":type,
    'Safety_Security_Package':safety_securty
}

df_input = pd.DataFrame.from_dict([my_dict])
st.header("The configuration of your car is below")
st.table(df_input)



# defining the function which will make the prediction using the data

filename = 'final_model_chosen'
model = pickle.load(open(filename, 'rb'))


if st.button("Predict"):
    pred = model.predict(df_input)
    st.success("The estimated price of your car is €{}. ".format(pred[0]))






