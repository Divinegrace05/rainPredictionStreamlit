import pickle
import streamlit as st
import pandas as pd
import numpy as np

@st.cache(suppress_st_warning=True)
def encode_location(location):
    locations = [
        'Airbase', 'BabaDogo', 'ClayCity', 'Embakasi', 'Karen', 'Karura',
        'NairobiCentral', 'NairobiSouth', 'NairobiWest', 'Ruai', 'SouthC',
        'UmojaI', 'UmojaIi', 'UpperSavannah', 'Utalii', 'Utawala',
        'UthiruRuthimitu', 'Njiru', 'Pangani', 'Ngara', 'Mwiki',
        'ParklandsHighridge', 'Pumwani', 'Riruta', 'Roysambu',
        'Sarangombe', 'Viwandani', 'DandoraAreaI', 'DandoraAreaIi',
        'DandoraAreaIii', 'DandoraAreaIv', 'EastleighNorth',
        'EastleighSouth', 'Gatina', 'Githurai', 'Harambee', 'Kahawa',
        'KahawaWest', 'KariobangiNorth', 'KariobangiSouth',
        'KayoleCentral', 'KayoleNorth', 'KayoleSouth', 'Kileleshwa',
        'WoodleyKenyattaGolfCourse', 'Waithaka', 'Zimmerman',
        'ZiwaniKariokor'
    ]
    
    # Handle unknown location
    if location not in locations:
        st.warning(f"Unknown location: {location}. Using default encoding.")
        return -1
    
    return locations.index(location)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def prediction(Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, 
        WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
        Cloud9am, Cloud3pm, Temp9am, Temp3pm, WindDir9am, WindDir3pm, WindGustDir, RainToday):
    

    # Preprocess categorical features
    encoded_location = encode_location(Location)
    wind_dir_9am_encoded = encode_location(WindDir9am)
    wind_dir_3pm_encoded = encode_location(WindDir3pm)
    wind_gust_dir_encoded = encode_location(WindGustDir)
    rain_today_encoded = 1 if RainToday == 'Yes' else 0

    input_data = np.array([
        Date, encoded_location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
        WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
        Cloud9am, Cloud3pm, Temp9am, Temp3pm, wind_dir_9am_encoded, wind_dir_3pm_encoded,
        wind_gust_dir_encoded, rain_today_encoded
    ]).astype(np.float64)
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
        WindDir9am = st.sidebar.selectbox("Wind Direction 9am", ("N", "E", "W", "S", "NW", "NE", "SW", "SE"))
        WindDir3pm = st.sidebar.selectbox("Wind Direction 3pm", ("E", "N", "W", "S", "NW", "NE", "SW", "SE"))
        WindGustDir = st.sidebar.selectbox("Wind Gust Direction", ("NW", "E", "N", "S", "W", "NE", "SW", "SE"))
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
        'WoodleyKenyattaGolfCourse', 'Waithaka', 'Zimmerman',
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

        if st.button("Predict"):
            # loading the trained model
            pickle_in = open('C:/Users/Han/xgboost.pkl', 'rb') 
            xgboost = pickle.load(pickle_in)


            input_data_reshaped = prediction(
                Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed,
                WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am,
                Cloud3pm, Temp9am, Temp3pm, WindDir9am, WindDir3pm, WindGustDir, RainToday
            ).reshape(1, -1)

            prediction_result = xgboost.predict(input_data_reshaped)

            if prediction_result[0] == 0:
                st.error('Tomorrow will be a bright sunny day!')
            elif prediction_result[0] == 1:
                st.success('Expect raindrops tomorrow!')

if __name__ == '__main__':
    main()