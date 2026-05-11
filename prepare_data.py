"""
Data Preparation Script for Accident Risk Prediction Model
====================================================
This script processes the Indian road accident dataset and prepares it for ML modeling.

Steps:
1. Loads accident.csv
2. Extracts hour from Time of Day column
3. Creates Is_Weekend column (1 for Saturday/Sunday, 0 otherwise)
4. Converts Severity to High_Risk (1 for Serious/Fatal, 0 for Minor)
5. Creates simulated Temperature column (value: 75)
6. Creates Visibility column based on Weather conditions
7. Saves cleaned dataset as accidents_clean.csv
"""

import pandas as pd

# ----------------------------------
# Step 1: Load the dataset
# ----------------------------------
print("Loading accident data...")
df = pd.read_csv("accident.csv")
print(f"Loaded {len(df)} records")

# ----------------------------------
# Step 2: Extract Hour from Time of Day
# ----------------------------------
# The "Time of Day" column contains times like "1:46", "21:30"
# We need to extract just the hour (first part before the colon)

def extract_hour(time_str):
    """Extract hour from time string like '1:46' or '21:30'"""
    try:
        # Split by colon and take the first part (hour)
        hour = int(str(time_str).split(':')[0])
        return hour
    except:
        return 0  # Default to 0 if parsing fails

df['Hour'] = df['Time of Day'].apply(extract_hour)
print(f"Extracted hours from Time of Day column")

# ----------------------------------
# Step 3: Create Is_Weekend column
# ----------------------------------
# 1 if Day is Saturday or Sunday, 0 otherwise

weekend_days = ['Saturday', 'Sunday']
df['Is_Weekend'] = df['Day of Week'].apply(
    lambda x: 1 if x in weekend_days else 0
)
print(f"Created Is_Weekend column")

# ----------------------------------
# Step 4: Convert Severity to High_Risk
# ----------------------------------
# 1 if Severity is "Serious" or "Fatal", 0 if "Minor"

high_risk_severity = ['Serious', 'Fatal']
df['High_Risk'] = df['Accident Severity'].apply(
    lambda x: 1 if x in high_risk_severity else 0
)
print(f"Created High_Risk column from Accident Severity")

# ----------------------------------
# Step 5: Create simulated Temperature column
# ----------------------------------
# Since there is no Temperature column, create one with value 75
df['Temperature'] = 75
print(f"Created Temperature column with value 75")

# ----------------------------------
# Step 6: Create Visibility column based on Weather
# ----------------------------------
# 3 if Weather is Foggy, Rainy, or Stormy
# 10 otherwise

low_visibility_weather = ['Foggy', 'Rainy', 'Stormy']
df['Visibility'] = df['Weather Conditions'].apply(
    lambda x: 3 if x in low_visibility_weather else 10
)
print(f"Created Visibility column based on Weather Conditions")

# ----------------------------------
# Step 7: Add Latitude and Longitude based on State
# ----------------------------------
# Dictionary of approximate coordinates for Indian states
state_coordinates = {
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2441, 92.5376),
    'Bihar': (25.0961, 85.3131),
    'Chandigarh': (30.7333, 76.7794),
    'Chhattisgarh': (21.2787, 81.8662),
    'Delhi': (28.6139, 77.2090),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2587, 71.1924),
    'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jammu and Kashmir': (33.7782, 76.5762),
    'Jharkhand': (23.6102, 85.2795),
    'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985),
    'Punjab': (31.1471, 75.3412),
    'Puducherry': (11.9416, 79.8083),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5129),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550),
}

def get_coordinates(state):
    """Get latitude and longitude for a state"""
    if state in state_coordinates:
        lat, lon = state_coordinates[state]
        # Add small random variation for visualization variety
        import random
        lat += random.uniform(-0.5, 0.5)
        lon += random.uniform(-0.5, 0.5)
        return lat, lon
    else:
        # Default to central India if state not found
        import random
        return 22.0 + random.uniform(-1, 1), 78.0 + random.uniform(-1, 1)

# Add coordinates based on State Name
df['Coordinates'] = df['State Name'].apply(get_coordinates)
df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])
print(f"Added Latitude and Longitude columns based on State")

# ----------------------------------
# Step 8: Keep only required columns
# ----------------------------------
columns_to_keep = ['Hour', 'Is_Weekend', 'Temperature', 'Visibility', 'High_Risk', 'Latitude', 'Longitude']
df_clean = df[columns_to_keep]
print(f"Selected required columns: {columns_to_keep}")

# ----------------------------------
# Step 8: Save the cleaned dataset
# ----------------------------------
output_file = "accidents_clean.csv"
df_clean.to_csv(output_file, index=False)
print(f"Saved cleaned dataset to {output_file}")

# Display summary
print("\n" + "="*50)
print("DATA PREPARATION COMPLETE!")
print("="*50)
print(f"Total records: {len(df_clean)}")
print(f"\nColumn summary:")
print(df_clean.info())
print(f"\nFirst few rows:")
print(df_clean.head())
print(f"\nHigh_Risk distribution:")
print(df_clean['High_Risk'].value_counts())
