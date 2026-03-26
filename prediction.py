import numpy as np
import joblib

model = joblib.load("house_price_model.pkl")

# ----- User Inputs -----
lon = float(input("Longitude: "))
lat = float(input("Latitude: "))
age = float(input("Housing Median Age: "))
rooms = float(input("Total Rooms: "))
bedrooms = float(input("Total Bedrooms: "))
population = float(input("Population: "))
households = float(input("Households: "))
income = float(input("Median Income: "))

location = input("Location (INLAND / <1H OCEAN / NEAR BAY / NEAR OCEAN / ISLAND): ").upper()

# ----- Apply same transforms -----
rooms_log = np.log(rooms + 1)
bedrooms_log = np.log(bedrooms + 1)
population_log = np.log(population + 1)
households_log = np.log(households + 1)

bedroom_ratio = bedrooms_log / rooms_log
household_rooms = rooms_log / households_log

# ----- One-hot encoding for location -----
loc_list = ["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"]
loc_vector = [1 if location == loc else 0 for loc in loc_list]

# ----- Final input array -----
row = np.array([[lon, lat, age, rooms_log, bedrooms_log, population_log, households_log,
                 income, bedroom_ratio, household_rooms] + loc_vector])

columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'bedroom_ratio', 'household_rooms',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
]

import pandas as pd

input_df = pd.DataFrame(row, columns=columns)
price = model.predict(input_df)[0]

indian_price = price * 83 * 0.35
print(f"\nPredicted House Price (Indian currency): ₹{indian_price:,.0f}")


