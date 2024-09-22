# Here I clean up the data that I scraped from the website and prepare the data for a machine learning model
import pandas as pd
import numpy as np
import folium  # folium is used for creating a map of the properties
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas.api.types as ptypes
from sqlalchemy import create_engine
import re
import time
import os
from dotenv import load_dotenv
import requests

# Read in the data saved in my SQL database
cnx = create_engine('sqlite:///carneige-property-db.db').connect()
df = pd.read_sql_table('property', cnx)


### Perform clean-up
# Clean up 'Price' column to extract only the house price
pattern = r'\$\d{1,3}(?:,\d{3})*'
df['price_clean'] = df['Price'].apply(lambda x: re.search(pattern, x).group(0).replace('$','').replace(',','') if re.search(pattern, x) else np.nan)

# Clean up building type (unit, apartment, house etc...)
pattern = r'Building Type(Apartment|Commercial|House|Townhouse|Unit|Unknown)'
# Apply the pattern to the 'Building info' column to extract the building type
df['building_type_clean'] = df['Building info'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else np.nan)

# Clean up to get year building was built
pattern = r'Year Built(\d{4})'
df['year_built_clean'] = df['Building info'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else np.nan)

# Clean up to get floor size
pattern = r'Floor Size(\d{2,4})m2'
df['floor_size_clean'] = df['Building info'].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else np.nan)

# Clean up address ('Carnegie' is fused with other text)
df['address_clean'] = df['Address'].str.replace(r'Carnegie VIC 3163',' Carnegie VIC 3163')

# Clean up description column (i.e. remove 'Last listing date')
df['description_clean'] = df['Description'].str.replace(r'Last Listing description \([A-Za-z]+ \d{4}\)','', regex=True)

# Clean up room, shower, car and size columns then rename these columns
df[['Room', 'Shower', 'Car', 'Size']] = df[['Room', 'Shower', 'Car', 'Size']].replace('NG', np.nan)

# Also rename those columns to indicate they are clean
rename_dict ={
    'Room':'room_clean',
    'Shower':'shower_clean',
    'Car':'car_clean',
    'Size':'size_clean'
}
df.rename(columns=rename_dict, inplace=True)

# Create a final 'clean' dataset
data_clean = df[['price_clean', 'building_type_clean',
                 'year_built_clean', 'floor_size_clean',
                 'address_clean','room_clean', 'shower_clean',
                 'car_clean', 'size_clean', 'description_clean']]

# Check clean dataset
print(data_clean.head())

data_clean.to_csv("melbourne-realestate-clean.csv", encoding='utf-8', index=False)
#%% Draw a Folium map to check that all the addresses are in the same proximity of each other 
## First, geoencode to get lat and longitude of all the addresses
# I use the positionstack API (https://positionstack.com/documentation) to get data from each address
# Load the API env variable and geoencode address
load_dotenv()  
positionstack_api_key = os.getenv('positionstack_api')

def geocode_address(address_input):
    """Get latitude and longitude of an input address using positionstack API. 
    Returns a tuple with lat, lon coordinates of address"""
    time.sleep(1)  # Sleep to stay under API rate limit
    url = 'http://api.positionstack.com/v1/forward'
    params = {'access_key': positionstack_api_key,
    'query': address_input,
    'limit': 1}
    resp = requests.get(url=url, params=params)
    data = resp.json()
    lat = data['data'][0]['latitude']
    lon = data['data'][0]['longitude']
    # Check progress
    print(f'Successfully processed address: {address_input}')
    return (lat, lon) if data['data'] else (None, None)

# Apply geocoding to the DataFrame
data_clean['coordinates'] = data_clean['address_clean'].apply(geocode_address)

# Save as a csv in case something goes wrong and we avoid running everything again 
data_clean.to_csv('raw-data/melb_real_estate_carnegie_map_coordinates.csv', encoding='utf-8', index=False)

## Draw map
# Initialize a map centered around some coordinates (e.g., the first valid location)
map_center = data_clean['coordinates'].iloc[0]
map_ = folium.Map(location=map_center, zoom_start=5)

# Add markers for each address
for i, row in data_clean.iterrows():
    folium.Marker(location=row['coordinates'], popup=row['address_clean']).add_to(map_)

# Save map to an HTML file
map_.save("images/map.html")

print(data_clean)