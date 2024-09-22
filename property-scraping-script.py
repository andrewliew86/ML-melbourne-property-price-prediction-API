# Scraping real-estate data

from urllib.request import urlopen as uReq
import urllib.request
from bs4 import BeautifulSoup as soup
import re
import pandas as pd
from tqdm import tqdm # gives you a progress bar as you download stuff in a for loop
import numpy as np
import time
import sqlite3

## First, we scrape all the links to the URLs for houses and units, we will then visit these links and scrape data from there
def scrape_links():
    '''Scrapes a website and saves the hrefs into a csv file'''
    url_list = []

    for i in tqdm(range(1,23)):
        # sleep is used to make sure that I dont spam the server too much
        time.sleep(2)
        my_url = "https://www.onthehouse.com.au/sold/vic/carnegie-3163?page={}".format(i)
        req = urllib.request.Request(my_url,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing
        page_soup = soup(page_html, 'html.parser')

        # get all url links
        try:
            for link in page_soup.find_all('a', attrs={'href': re.compile("/property/vic/carnegie-3163/.+")}): 
                url_list.append(link.get('href'))
                # display the actual urls 
                print(link.get('href'))
        except:
            continue

    # Add wwww to the fron of the urls that are scraped and save as a csv        
    final_url_list = [f"https://www.onthehouse.com.au{u}" for u in url_list]        
    df = pd.DataFrame(final_url_list, columns=['URLs'])
    df.drop_duplicates(inplace=True)
    df.to_csv("raw-data/url_list.csv")



## Step 2: Scrape data from each webpage
def scrape_data():
    # Load urls we scraped earlier
    url_data = pd.read_csv("url_list.csv")
    url_list = url_data['URLs'].to_list()

    # Create lists to store data
    price_list = []
    address_list = []
    build_list = []
    room_list = []
    shower_list = []
    car_list = []
    size_list = []
    description_list = []

    # loop through url list
    for url in tqdm(url_list):
        # sleep is used to make sure that I dont spam the server too much
        time.sleep(2)
        print(f"Processing url: {url}")
        req = urllib.request.Request(url ,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing
        page_soup = soup(page_html, 'html.parser')

        # get price
        try:
            price_container = page_soup.find_all(class_="lgText text-left PropertyInfo__propertyDisplayPrice--7k6yF")
            price = price_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
            price_list.append(price)
        except IndexError:
                price_list.append('NG')

        # get Address
        try:
            address_container = page_soup.find_all(class_="mb-1 mb-md-3 bold500 PropertyInfo__propertySuburbAddress--3q3f7")
            address = address_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
            address_list.append(address)
        except IndexError:
                address_list.append('NG')


        # Rooms, showers, car spaces and size of houses are all in a single container
        att_container = page_soup.find_all(class_="d-flex mb-2 flex-wrap PropertyInfo__propertyAttributes--TgDyq")
        room_bath_car_data = att_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")

        # Number of rooms
        try:
            rooms = re.search(r'Bedrooms:\s*(\d+)', room_bath_car_data).group(1) if re.search(r'Bedrooms:\s*(\d+)', room_bath_car_data) else 'NG'
            room_list.append(rooms)
        except:
            room_list.append('NG')
            
        # Number of showers
        try:
            baths = re.search(r'Bathrooms:\s*(\d+)', room_bath_car_data).group(1) if re.search(r'Bathrooms:\s*(\d+)', room_bath_car_data) else 'NG'
            shower_list.append(baths)
        except:
            shower_list.append('NG')
        
        # Number of car spaces
        try:
            cars = re.search(r'Car spaces:\s*(\d+)', room_bath_car_data).group(1) if re.search(r'Car spaces:\s*(\d+)', room_bath_car_data) else 'NG'
            car_list.append(cars)
        except:
            car_list.append('NG')      

        # Size
        try:
            size = re.search(r'Land area:\s*(\d+)', room_bath_car_data).group(1) if re.search(r'Land area:\s*(\d+)', room_bath_car_data) else 'NG'
            size_list.append(size)
        except:
            size_list.append('NG') 

        # building_info
        try:
            build_container = page_soup.find_all(class_="row m-0")
            build = build_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
            build_list.append(build)
        except IndexError:
                build_list.append('NG')

        # description_list
        try:
            description_container = page_soup.find_all(class_="mt-4 mb-4")
            description = description_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
            description_list.append(description)
        except IndexError:
                description_list.append('NG')


    # Make a dataframe.. Note that all the lists have to be the same length to create the dataframe!!
    df = pd.DataFrame(np.column_stack([price_list, address_list, build_list, 
                                    room_list, shower_list, 
                                    car_list, size_list, description_list]),
                                    columns=['Price', 'Address', 
                                                'Building info', 'Room', 
                                                'Shower', 'Car','Size',
                                                'Description'])


    df.to_csv('raw-data/melb_real_estate_carnegie_surrounding_detailed_19April24.csv', encoding='utf-8', index=False)


# Insert scraped data into a SQL database for more permanent storage
def insert_data_to_sql():
    # Sample data
    data = pd.read_csv('melb_real_estate_carnegie_surrounding_detailed_19April24.csv')

    # Create a connection to SQLite (or connect to an existing database)
    conn = sqlite3.connect('carneige-property-db.db')

    # Insert the DataFrame into the database
    data.to_sql('property', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()


#scrape_links()
#scrape_data()
insert_data_to_sql()
