import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import geopandas as gpd
# Load the dataset to encode the state names
file_path = 'states data.csv'
state_data = pd.read_csv(file_path)

state_data.columns = state_data.columns.str.strip()
# Example of cleaning data for 'Per Capita Income' and converting it to numeric
state_data['Per Capita Income'] = state_data['Per Capita Income'].replace({',': ''}, regex=True).astype(float)

# Repeat for other columns if needed
state_data['GDP'] = state_data['GDP'].replace({',': ''}, regex=True).astype(float)
state_data['Population (2024)'] = state_data['Population (2024)'].astype(str).str.replace(r'\D', '', regex=True).astype(float)
state_data['Literacy Rate'] = state_data['Literacy Rate'].str.rstrip('%').astype(float)
# Convert columns to numeric if necessary
state_data['Mortality Rate'] = pd.to_numeric(state_data['Mortality Rate'], errors='coerce')
state_data['Birth rate'] = pd.to_numeric(state_data['Birth rate'], errors='coerce')
state_data['Life Expectancy'] = pd.to_numeric(state_data['Life Expectancy'], errors='coerce')

state_data['Economic_Score'] = ( 0.12 * state_data['Per Capita Income'] +
                         0.70 * state_data['GDP'] +
                         0.03 * state_data['Population (2024)'] +
                         0.01 * state_data['Life Expectancy'] +
                         0.01 * (100 -state_data['Birth rate']) +
                         0.12 * state_data['Literacy Rate'] +
                         0.01 * (100 - state_data['Mortality Rate']))

# Encode the state names
label_encoder = LabelEncoder()
state_data['State_Encoded'] = label_encoder.fit_transform(state_data['State'])
# Load the trained model
model_filename = 'trained_model2.pkl'
loaded_model = joblib.load(model_filename)

# Scale the feature
scaler = StandardScaler()
scaler.fit(state_data[['State_Encoded']]) 

# GUI Function to Predict Economic Status
def predict_economic_status():
    state_name = state_var.get()
    if state_name not in state_data['State'].values:
        messagebox.showerror("Error", "Invalid State Name")
        return
    # Get encoded state and GDP per capita
    state_encoded = label_encoder.transform([state_name])[0]
    
    # Scale the features for prediction
    input_features = scaler.transform([[state_encoded]])
    
    # Make prediction
    predicted_values = loaded_model.predict(input_features)
    economic_status = pd.DataFrame(predicted_values, columns=state_data.drop(columns=['State', 'State_Encoded']).columns)
    
    # Display prediction results
    messagebox.showinfo("Predicted Economic Status", economic_status.to_string(index=False))

# Create GUI
root = tk.Tk()
root.title("Economic Status Predictor")

# State Input
tk.Label(root, text="Enter State Name:").grid(row=0, column=0, padx=10, pady=10)
state_var = tk.StringVar()
state_entry = ttk.Entry(root, textvariable=state_var)
state_entry.grid(row=0, column=1, padx=10, pady=10)
# Predict Button
predict_button = ttk.Button(root, text="Predict Economic Status", command=predict_economic_status)
predict_button.grid(row=1, columnspan=2, pady=10)

root.mainloop()