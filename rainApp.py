import pickle
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

@st.cache_data()
def encode_location(location):
    locations = ['Airbase', 'BabaDogo', 'ClayCity', 'Embakasi', 'Karen', 'Karura', 'NairobiCentral', 
    'NairobiSouth', 'NairobiWest', 'Ruai', 'SouthC', 'UmojaI', 'UmojaIi', 'UpperSavannah', 'Utalii', 
    'Utawala', 'UthiruRuthimitu', 'Njiru', 'Pangani', 'Ngara', 'Mwiki', 'ParklandsHighridge', 'Pumwani', 
    'Riruta', 'Roysambu', 'Sarangombe', 'Viwandani', 'DandoraAreaI', 'DandoraAreaIi', 'DandoraAreaIii', 
    'DandoraAreaIv', 'EastleighNorth', 'EastleighSouth', 'Gatina', 'Githurai', 'Harambee', 'Kahawa', 
    'KahawaWest', 'KariobangiNorth', 'KariobangiSouth', 'KayoleCentral', 'KayoleNorth', 'KayoleSouth', 
    'Kileleshwa', 'WoodleyKenyattaGolfCourse', 'Waithaka', 'Zimmerman', 'ZiwaniKariokor']

    encoded_location = [0] * len(locations)
    try:
        index = locations.index(location)
        encoded_location[index] = 1
    except ValueError:
        pass  # Handle the case if location is not found in the list
    return encoded_location

@st.cache_data()
def preprocess_date(date):
    reference_date = dt.date(1900, 1, 1)
    return (date - reference_date).days

@st.cache_data()
def encode_categorical_variables(wind_dir_9am, wind_dir_3pm, wind_gust_dir, rain_today):
    # Encode wind directions
    wind_dir_mapping = {'W': 0, 'WNW':1, 'WSW':2, 'NE':3, 'NNW':4, 'N':5, 'NNE':6, 'SW':7, 'ENE':8,
       'SSE':9, 'S':10, 'NW':11, 'SE':12, 'ESE':13, 'E':14, 'SSW':15}
    encoded_wind_dir_9am = wind_dir_mapping.get(wind_dir_9am, -1)
    encoded_wind_dir_3pm = wind_dir_mapping.get(wind_dir_3pm, -1)
    encoded_wind_gust_dir = wind_dir_mapping.get(wind_gust_dir, -1)
    
    # Encode RainToday
    encoded_rain_today = 1 if rain_today == 'Yes' else 0
    
    return encoded_wind_dir_9am, encoded_wind_dir_3pm, encoded_wind_gust_dir, encoded_rain_today

@st.cache_data()
def prediction(Date, encoded_location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, 
        WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
        Cloud9am, Cloud3pm, Temp9am, Temp3pm, WindDir9am, WindDir3pm, WindGustDir, RainToday):
    
    # Convert Date to numerical representation
    processed_date = preprocess_date(Date)
    
    # Encode categorical variables
    encoded_wind_dir_9am, encoded_wind_dir_3pm, encoded_wind_gust_dir, encoded_rain_today = encode_categorical_variables(
        WindDir9am, WindDir3pm, WindGustDir, RainToday)
    
    # Construct input_data array with exactly 24 features
    input_data = np.array([
        processed_date, *encoded_location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
        WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
        Cloud3pm, Temp9am, Temp3pm, encoded_wind_dir_9am, encoded_wind_dir_3pm, encoded_wind_gust_dir, encoded_rain_today
    ], dtype=np.float64)[:24]  # Select the first 24 features
    
    return input_data

def main():
    app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

    if app_mode=='Home':    
        st.title('RAIN PREDICTION ML APP:')      
        st.image('weather-forecast.jpg')
        data = pd.read_csv('weatherDataset.csv')
        st.write(data.head())
        st.bar_chart(data[['Rainfall', 'MinTemp']].head(20))
        st.line_chart(data[['Temp9am', 'Humidity9am']].head(20))

    elif app_mode == 'Prediction':    
        st.subheader('Fill ALL necessary information in order to know whether it will rain tomorrow or not !')    
            
        WindGustSpeed=st.sidebar.text_input('Wind Gust Speed', value= '0')
        Evaporation=st.sidebar.slider('Evaporation',min_value=0,max_value=30,value=7,step=1)
        Date = st.sidebar.date_input('Date')
        WindDir9am = st.sidebar.selectbox("Wind Direction 9am", ('W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
        WindDir3pm = st.sidebar.selectbox("Wind Direction 3pm", ('W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
        WindGustDir = st.sidebar.selectbox("Wind Gust Direction", ('W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'))
        RainToday = st.sidebar.radio("Did it Rain Today?", ("No", "Yes"))
        Location = st.sidebar.selectbox("Location", ('Airbase', 'BabaDogo', 'ClayCity', 'Embakasi', 'Karen', 'Karura',
        'NairobiCentral', 'NairobiSouth', 'NairobiWest', 'Ruai', 'SouthC',
        'UmojaI', 'UmojaIi', 'UpperSavannah', 'Utalii', 'Utawala',
        'UthiruRuthimitu', 'Njiru', 'Pangani', 'Ngara', 'Mwiki',
        'ParklandsHighridge', 'Pumwani', 'Riruta', 'Roysambu',
        'Sarangombe', 'Viwandani', 'DandoraAreaI', 'DandoraAreaIi',
        'DandoraAreaIii', 'DandoraAreaIv', 'EastleighNorth',
        'EastleighSouth', 'Gatina', 'Githurai', 'Harambee', 'Kahawa',
        'KahawaWest', 'KariobangiNorth', 'KariobangiSouth',
        'KayoleCentral', 'KayoleNorth', 'KayoleSouth', 'Kileleshwa',
        'WoodleyKenyattaGolf', 'KayoleSouth', 'Kileleshwa', 'WoodleyKenyattaGolfCourse', 'Waithaka', 'Zimmerman',
        'ZiwaniKariokor' ))

        col1,col2 = st.columns(2)
        with col1:
            Sunshine=st.slider('Sunshine',min_value=0,max_value=30,value=12,step=1)
            MaxTemp=st.slider('Maximum Temperature(Celsius)',min_value=0,max_value=60,value=35,step=1)
            WindSpeed9am=st.slider('Wind Speed 9am Speed',min_value=0,max_value=40,value=11,step=1)
            Humidity9am=st.number_input('Humidity 9am',min_value=0, max_value=100, value=89, step=1)
            Pressure9am=st.number_input('Pressure 9am',min_value=0,max_value=2000,value=1010, step=1)
            Cloud9am = st.number_input('Cloud 9am',min_value=0,max_value=100,value=8, step=1)
            Temp9am = st.number_input('Temperature 9am',min_value=0,max_value=40,value=20, step=1)
                
        with col2:
            Rainfall=st.slider('Rainfall',min_value=0,max_value=60,value=2,step=1)
            MinTemp=st.slider('Minimum Temperature(Celsius)',min_value=0,max_value=40,value=7,step=1)
            WindSpeed3pm=st.slider('Wind Speed 3pm',min_value=0,max_value=40,value=9,step=1)
            Humidity3pm=st.number_input('Humidity 3pm',min_value=0,max_value=100,value=15, step=1)
            Pressure3pm=st.number_input('Pressure 3pm',min_value=0,max_value=2000,value=993, step=1)
            Cloud3pm = st.number_input('Cloud 3pm',min_value=0,max_value=100,value=9, step=1)
            Temp3pm = st.number_input('Temperature 3pm',min_value=0,max_value=40,value=12, step=1)

        encoded_location = encode_location(Location)

        if st.button("Predict"):
            # loading the trained model
            model = pickle.load(open("C:/Users/Han/xgb.pkl", "rb"))

            input_data = prediction(
                Date, encoded_location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed,
                WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
                Cloud3pm, Temp9am, Temp3pm, WindDir9am, WindDir3pm, WindGustDir, RainToday
            ).reshape(1, -1)

            prediction_result = model.predict(input_data)
            print("Prediction Result:", prediction_result)

            if prediction_result[0] == 0:
                st.error('It will not rain tommorrow!')
                st.image('sunny.gif', caption= 'Developed with ❤ by Grace Gitau')
            elif prediction_result[0] == 1:
                st.success('It will rain tomorrow. Dont forget to carry an umbrella!')
                st.image('raining.gif', caption= 'Developed with ❤ by Grace Gitau')

            
if __name__ == '__main__':
    main()

