import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

feature_names = [
    'gender', 'age', 'customer type', 'type of travel', 'class',
    'flight distance', 'departure delay in minutes', 'arrival delay in minutes',
    'inflight wifi service', 'departure/arrival time convenient',
    'ease of online booking', 'gate location', 'food and drink',
    'online boarding', 'seat comfort', 'inflight entertainment',
    'on-board service', 'leg room service', 'baggage handling',
    'checkin service', 'inflight service', 'cleanliness'
]

label_encoding = {
    'gender' : {'Female' : 0, 'Male' : 1},
    'customer type' : {'Loyal Customer' : 0 ,'disloyal Customer' : 1},
    'class' : {'Business' : 0, 'Eco' : 1, 'Eco Plus' : 2},
    'type of travel' : {'Business travel' : 0, 'Personal Travel' : 1},
}

def preprocess_input(data):
    df = pd.DataFrame([data], columns=feature_names)
    for column in label_encoding:
        df[column] = df[column].map(label_encoding[column])

    numeric_columns = ['age', 'flight distance', 'departure delay in minutes',
       'arrival delay in minutes', 'inflight wifi service',
       'departure/arrival time convenient', 'ease of online booking',
       'gate location', 'food and drink', 'online boarding', 'seat comfort',
       'inflight entertainment', 'on-board service', 'leg room service',
       'baggage handling', 'checkin service', 'inflight service',
       'cleanliness']
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

#Перехожу к вебу

service_polls = {
    'inflight wifi service' : 'Качество wi-fi на борту',
    'departure/arrival time convenient' : 'Удобство времени вылета/прилета',
    'ease of online booking' : 'Легкость онлайн-бронировавния',
    'gate location' : 'Расположение выхода на посадку',
    'food and drink' : 'Качество еди и напитков во время полета',
    'online boarding' : 'Удобство онлайн-регистрации',
    'seat comfort' : 'Удобство кресел',
    'inflight entertainment' : 'Развлечения в полете',
    'leg room service' : 'Пространство для ног',
    'baggage handling' : 'Перевоз багажа',
    'checkin service' : 'Обсуживание во время регистрации',
    'inflight service' : 'Обслуживание на борту',
    'on-board service' : 'Оценка полета',
    'cleanliness' : 'Чистота в самолете'
}

st.title('Прогноз удовлетворенности авиапассажиров ✈️')
st.write('Оцените полет по различным показателям')

with st.form('prediction_form'):
    gender = st.selectbox('Пол', ['Женский', 'Мужской'])
    age = st.number_input('Возраст', min_value = 7, max_value = 104, value = 35)

    customer_type_display = st.selectbox('Ваша категория', ['Обычный пассажир', 'В системе лояльности'])
    customer_mapping = {
        'Обычный пассажир': 'disloyal Customer',
        'В системе лояльности': 'Loyal Customer'
    }
    customer_type = customer_mapping[customer_type_display]

    type_of_travel_display = st.selectbox('Цель поездки', ['Бизнес поездка', 'Личная поездка'])
    type_of_travel_mapping = {
        'Бизнес поездка' : 'Business travel',
        'Личная поездка' : 'Personal Travel'
    }
    type_of_travel = type_of_travel_mapping[type_of_travel_display]

    flight_class_display = st.selectbox('Класс', ['Эко', 'Эко Плюс', 'Бизнес'])
    flight_class_mapping = {
        'Эко' : 'Eco',
        'Эко Плюс' : 'Eco Plus',
        'Бизнес' : 'Business'
    }
    flight_class = flight_class_mapping[flight_class_display]

    flight_distance = st.number_input('Дистанция полета (км)', min_value=30, max_value=6500, value = 750)
    departure_delay_in_minutes = st.number_input('Задержка вылета в минутах', min_value=0, max_value=1500)
    arrival_delay_in_minutes = st.number_input('Задержка прибытия в минутах', min_value=0, max_value=1500)

    st.subheader('Оцените сервисы (1 - очень плохо, 5 - отлично)')
    service_rating = {}
    for service_key, service_label in service_polls.items():
        rating = st.radio(
            service_label,
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True
        )
        service_rating[service_key] = rating

    submitted = st.form_submit_button('Предсказать удовлетворенность полетом')

if submitted:
    input_data = {
        'gender' : gender,
        'age' : age,
        'customer type' : customer_type,
        'type of travel' : type_of_travel,
        'class' : flight_class,
        'flight distance' : flight_distance,
        'departure delay in minutes' : departure_delay_in_minutes,
        'arrival delay in minutes' : arrival_delay_in_minutes,
    }

    input_data.update(service_rating)

    gender_mapping = {
        'Женский' : 'Female',
        'Мужской' : 'Male'
    }
    input_data['gender'] = gender_mapping[input_data['gender']]

    input_df = preprocess_input(input_data)

    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Результат прогноза')
        if prediction[0] == 1:
            st.success(f"✅ Пассажир доволен полётом (вероятность: {prediction_proba[0][1] * 100:.2f}%)")
        else:
            st.error(f"❌ Пассажир недоволен полётом (вероятность: {prediction_proba[0][0] * 100:.2f}%)")


        with st.expander('Детали введенных данных'):
            st.write('### Оценки сервисов')
            for service, rating in service_rating.items():
                st.write(f"{service_polls[service]}: {'⭐' * rating} ({rating}/5)")

            st.write('### Другие параметры')
            st.json({
                'Возраст': age,
                'Тип клиента': customer_type_display,
                'Цель поездки': type_of_travel_display,
                'Класс': flight_class_display,
                'Дистанция полёта (км)': flight_distance,
                'Задержка вылета (мин)': departure_delay_in_minutes,
                'Задержка прибытия (мин)': arrival_delay_in_minutes
            })

    except Exception as e:
        st.error(f"Произошла ошибка при предсказании: {str(e)}")
