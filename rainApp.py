import pickle
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

@st.cache_data()
def encode_categorical_variables(wind_dir_9am, rain_today):
    # Encode wind directions
    wind_dir_mapping = {'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7,
    'S': 8, 'SSW': 9, 'SW': 10, 'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15}
    encoded_wind_dir_9am = wind_dir_mapping.get(wind_dir_9am, -1)
    
    # Encode RainToday
    encoded_rain_today = 1 if rain_today == 'Yes' else 0
    
    return encoded_wind_dir_9am, encoded_rain_today

@st.cache_data()
def prediction(MinTemp, MaxTemp, Sunshine, Rainfall, WindGustSpeed, WindDir9am, WindSpeed9am, WindSpeed3pm, 
            Humidity9am, Humidity3pm, Cloud9am, Cloud3pm, RainToday, Date, Location):
       
    # Encode categorical variables    
    encoded_wind_dir_9am, encoded_rain_today = encode_categorical_variables(
        wind_dir_9am=WindDir9am, rain_today=RainToday)
    
    # Construct input_data array with exactly 12 features   
    input_data = np.array([  MinTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity3pm,
        Humidity9am, Cloud9am, Cloud3pm, encoded_rain_today, encoded_wind_dir_9am  ], dtype=np.float64) 
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
        
        Date = st.sidebar.date_input('Date', min_value=dt.date(2008, 1, 1), max_value=dt.date.today(), value=dt.date(2018, 12, 5))
        Location = st.sidebar.selectbox("Location", ('Airbase', 'BabaDogo', 'ClayCity', 'Embakasi', 'Karen', 'Karura', 'NairobiCentral', 'NairobiSouth', 
        'NairobiWest', 'Ruai', 'SouthC', 'UmojaI', 'UmojaIi', 'UpperSavannah', 'Utalii', 'Utawala', 'UthiruRuthimitu', 'Njiru', 'Pangani', 'Ngara', 
        'Mwiki', 'ParklandsHighridge', 'Pumwani', 'Riruta', 'Roysambu', 'Sarangombe', 'Viwandani', 'DandoraAreaI', 'DandoraAreaIi', 'DandoraAreaIii', 
        'DandoraAreaIv', 'EastleighNorth','EastleighSouth', 'Gatina', 'Githurai', 'Harambee', 'Kahawa', 'KahawaWest', 'KariobangiNorth', 'KariobangiSouth',
        'KayoleCentral', 'KayoleNorth', 'KayoleSouth', 'Kileleshwa', 'WoodleyKenyattaGolf', 'KayoleSouth', 'Kileleshwa', 'WoodleyKenyattaGolfCourse', 
        'Waithaka', 'Zimmerman', 'ZiwaniKariokor' ))
        WindDir9am = st.sidebar.selectbox("Wind Direction 9am", ('ENE', 'NNW', 'W', 'NE', 'SW', 'SSE', 'S', 'SE', 'SSW', 'N', 'WSW', 'ESE', 'E', 'NW',
                           'WNW', 'NNE'))    
        WindGustSpeed= int(st.sidebar.text_input('Wind Gust Speed', value= '41'))
        RainToday = st.sidebar.radio("Did it Rain Today?", ("No", "Yes"))        

        col1,col2 = st.columns(2)
        with col1:
            Rainfall=st.slider('Rainfall',min_value=0.0,max_value=60.0,value=0.0,step=0.1)
            MaxTemp = st.slider('Maximum Temperature(Celsius)',min_value=0,max_value=60,value=12,step=1)
            WindSpeed9am=st.slider('Wind Speed 9am' ,min_value=0,max_value=40,value=1,step=1)
            Humidity9am=st.number_input('Humidity 9am',min_value=0, max_value=100, value=23, step=1)
            Cloud9am = st.number_input('Cloud 9am',min_value=0,max_value=100,value=0, step=1)            
                
        with col2:       
            Sunshine=st.slider('Sunshine',min_value=0.0,max_value=15.0,value=9.7,step=0.1)     
            MinTemp=st.slider('Minimum Temperature(Celsius)',min_value=0,max_value=60,value=0,step=1)
            WindSpeed3pm=st.slider('Wind Speed 3pm',min_value=0,max_value=40,value=25,step=1)
            Humidity3pm=st.number_input('Humidity 3pm',min_value=0,max_value=100,value=0, step=1)
            Cloud3pm = st.number_input('Cloud 3pm',min_value=0,max_value=100,value=9, step=1)

        if st.button("Predict"):
            # loading the trained model
            model = pickle.load(open("C:/Users/Han/xgb.pkl", "rb"))

            input_data = prediction(MinTemp, MaxTemp, Sunshine, Rainfall, WindGustSpeed, WindDir9am, WindSpeed9am, 
            WindSpeed3pm, Humidity9am, Humidity3pm, Cloud9am, Cloud3pm, RainToday, Date, Location).reshape(1, -1)

            prediction_result = model.predict(input_data)
            print("Prediction Result:", prediction_result)

            if prediction_result[0] == 0:
                st.error('It will not rain tommorrow!')
                st.image('sunny.gif', caption= 'Developed with ❤ by Grace Gitau')             
            else:
                st.success('It will rain tomorrow. Dont forget to carry an umbrella!')
                st.image('raining.gif', caption= 'Developed with ❤ by Grace Gitau') 

if __name__ == '__main__':
    main()

