import pickle
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import firebase_admin
from firebase_admin import credentials, auth
import requests
import plotly.express as px

# Check if Firebase app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase app using service account credentials
    cred = credentials.Certificate("D:/Downloads/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

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

# Create Account (Sign Up)
def sign_up(email, password):
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return user.uid
    except Exception as e:
        st.error(f"Error creating account: {e}")

def log_in(email, password):
    try:
        # Make a POST request to Firebase REST API for user authentication
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCr8npYNB3ULaX9u6gujdpguVVevI9Ei0k",
            json={"email": email, "password": password, "returnSecureToken": True}
        )
        
        if response.status_code == 200:
            user_data = response.json()
            return user_data['localId']
        else:
            error_message = response.json().get('error', {}).get('message', 'Unknown error')
            st.error(f"Authentication failed: {error_message}")
            return None
    except Exception as e:
        # Handle other exceptions
        error_message = str(e)
        st.error(f"An unexpected error occurred: {error_message}")
        return None
            
# Forgot Password
def forgot_password(email):
    try:
        # Send password reset email using Firebase Admin SDK
        auth.generate_password_reset_link(email)
        st.success("Password reset email sent. Check your inbox.")
    except Exception as e:
        st.error(f"Error sending password reset email: {e}")

# Function to retrieve latest weather data for a specific location from the database
def get_latest_weather_data(location):
    # Replace this with your actual implementation to fetch data from your database
    # Example:
    # Load data from your database
    data = pd.read_csv('weatherDataset.csv')
    # Filter data for the selected location and get the latest row
    latest_weather = data[data['Location'] == location].iloc[-1]
    return latest_weather

# Streamlit UI
def main():
    
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    if st.session_state["user_id"]:
        # User is logged in
        page = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Logout"])

        if page == "Home":
            st.title('RAIN PREDICTION ML APP:')      
            st.image('weather-forecast.jpg')

            # Latest Weather Section
            st.header("Latest Weather Information")
            location = st.selectbox("Choose a location:", ['Airbase', 'BabaDogo', 'ClayCity', 'Embakasi', 'Karen', 'Karura', 'NairobiCentral', 'NairobiSouth', 
                                                    'NairobiWest', 'Ruai', 'SouthC', 'UmojaI', 'UmojaIi', 'UpperSavannah', 'Utalii', 'Utawala', 'UthiruRuthimitu', 
                                                    'Njiru', 'Pangani', 'Ngara', 'Mwiki', 'ParklandsHighridge', 'Pumwani', 'Riruta', 'Roysambu', 'Sarangombe', 
                                                    'Viwandani', 'DandoraAreaI', 'DandoraAreaIi', 'DandoraAreaIii', 'DandoraAreaIv', 'EastleighNorth', 'EastleighSouth', 
                                                    'Gatina', 'Githurai', 'Harambee', 'Kahawa', 'KahawaWest', 'KariobangiNorth', 'KariobangiSouth', 'KayoleCentral', 
                                                    'KayoleNorth', 'KayoleSouth', 'Kileleshwa', 'WoodleyKenyattaGolf', 'KayoleSouth', 'Kileleshwa', 
                                                    'WoodleyKenyattaGolfCourse', 'Waithaka', 'Zimmerman', 'ZiwaniKariokor'])

            if st.button("Get Latest Weather"):
                latest_weather = get_latest_weather_data(location)
                st.subheader("Latest Weather Information for " + location)
                col1,col2, col3 = st.columns(3)
                with col1:                    
                    st.write("Min Temperature:", latest_weather['MinTemp'])
                    st.write("Max Temperature:", latest_weather['MaxTemp'])
                    st.write("Rainfall:", latest_weather['Rainfall'])
                    
                with col2:
                    st.write("Wind Gust Speed:", latest_weather['WindGustSpeed'])
                    st.write("Wind Direction 9am:", latest_weather['WindDir9am'])
                    st.write("Wind Direction 3pm:", latest_weather['WindDir3pm'])
                    st.write("Wind Speed 9am:", latest_weather['WindSpeed9am'])
                with col3:
                    st.write("Wind Speed 3pm:", latest_weather['WindSpeed3pm'])
                    st.write("Humidity 9am:", latest_weather['Humidity9am'])
                    st.write("Humidity 3pm:", latest_weather['Humidity3pm'])
                    st.write("Rain Today:", latest_weather['RainToday'])
                   
            st.header("Data Visualization")
            data = pd.read_csv('weatherDataset.csv')
            st.write(data.head())
            st.bar_chart(data[['Rainfall', 'MinTemp']].head(20))

        elif page == "Prediction":
            st.subheader('Fill ALL necessary information in order to know whether it will rain tomorrow or not !')    
     
            Date = st.sidebar.date_input('Date', min_value=dt.date(2008, 1, 1), max_value=dt.date(2030, 12, 31), value=dt.date(2024, 4, 6))
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
                    st.error('It will not rain tomorrow!')
                else:
                    st.success('It will rain tomorrow. Don\'t forget to carry an umbrella!')

                # Visualize prediction results
                prediction_df = pd.DataFrame({
                    'Prediction': ['No Rain', 'Rain'],
                    'Probability': [1 - prediction_result[0], prediction_result[0]]
                })
                
                fig = px.bar(prediction_df, x='Prediction', y='Probability', 
                            color='Prediction', labels={'Probability': 'Probability (%)'}, 
                            title='Prediction Result', text='Probability', 
                            color_discrete_map={'No Rain': 'brown', 'Rain': 'green'})
                
                st.plotly_chart(fig)

        elif page == "Logout":
            # Clear user session
            st.session_state["user_id"] = None
            st.success("Logged out successfully!")

            # Wait for 2 seconds
            time.sleep(2)

            # Redirect to authentication page
            st.experimental_rerun()

    else:
        # User is not logged in
        page = st.sidebar.radio("User Authentication", ["Log In", "Sign Up"])

        if page == "Log In":
            st.header("Log In")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            # Checkbox for "Forgot Password?"
            forgot_pw = st.checkbox("Forgot Password")      
            if forgot_pw:
                email_for_reset = email
                print("Send Reset Email button clicked")
                if st.button("Send Reset Email"):
                    forgot_password(email_for_reset)

            login_button = st.button("Log In")
            if login_button:  # Check if the login button is clicked
                if email and password:  # Check if both email and password are filled
                    user_id = log_in(email, password)
                    if user_id:
                        st.session_state["user_id"] = user_id
                        st.success("Logged in successfully!")
                        time.sleep(1)

                        # Redirect to next page
                        st.experimental_rerun()
                else:
                    st.warning("Please enter both email and password.")


        elif page == "Sign Up":
            st.header("Create Account")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            repeat_password = st.text_input("Confirm Password", type="password")
            if st.button("Sign Up") and password == repeat_password:
                user_id = sign_up(email, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                    st.success("Account created successfully!")
                    time.sleep(1)

if __name__ == "__main__":
    main()
