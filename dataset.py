import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

soil_moisture = np.random.randint(10,80,rows)
temperature = np.random.randint(20,40,rows)
humidity = np.random.randint(30,90,rows)
rainfall = np.random.randint(0,30,rows)

irrigation = []

for i in range(rows):
    if soil_moisture[i] < 35 and rainfall[i] < 10:
        irrigation.append(1)
    else:
        irrigation.append(0)

df = pd.DataFrame({
    "Soil_Moisture":soil_moisture,
    "Temperature":temperature,
    "Humidity":humidity,
    "Rainfall":rainfall,
    "Irrigation":irrigation
})

df.to_csv("irrigation_data.csv",index=False)

print("Dataset Created")