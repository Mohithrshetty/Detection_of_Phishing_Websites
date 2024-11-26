from gevent import monkey

monkey.patch_all(thread=False, select=False)

import requests as re
import grequests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import warnings
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe
import os
import time

disable_warnings(InsecureRequestWarning)
warnings.filterwarnings("ignore", category=re.packages.urllib3.exceptions.InsecureRequestWarning)

# File to keep track of begin and end
STATE_FILE = "scraping_state.txt"
URL_FILE_NAME = "top-1m.csv"
OUTPUT_FILE = os.path.join(os.getcwd(), "structured_data_legitimate.csv")

# Step 1: Read CSV to DataFrame
data_frame = pd.read_csv(URL_FILE_NAME)
URL_list = data_frame['url'].to_list()


# Read the current state from file
def read_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            begin, end = map(int, f.read().strip().split(','))
    else:
        begin, end = 1, 100
    return begin, end


# Write the current state to file
def write_state(begin, end):
    with open(STATE_FILE, 'w') as f:
        f.write(f"{begin},{end}")


# Function to scrape the content of the URL and convert to a structured form for each
def create_structured_data(url_list):
    data_list = []
    for i in range(len(url_list)):
        try:
            response = re.get(url_list[i], verify=False, timeout=4)
            if response.status_code != 200:
                print(i, ". HTTP connection was not successful for the URL: ", url_list[i])
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                data_list.append(vector)
        except re.exceptions.RequestException as e:
            print(i, " --> ", e)
            continue
    return data_list


# Main function to perform scraping
def scrape_and_save():
    while True:
        begin, end = read_state()
        if begin >= len(URL_list):
            print("Scraping complete.")
            break

        collection_list = URL_list[begin:end]
        tag = "http://"
        collection_list = [tag + url for url in collection_list]

        data = create_structured_data(collection_list)

        columns = [
            'has_title',
            'has_input',
            'has_button',
            'has_image',
            'has_submit',
            'has_link',
            'has_password',
            'has_email_input',
            'has_hidden_element',
            'has_audio',
            'has_video',
            'number_of_inputs',
            'number_of_buttons',
            'number_of_images',
            'number_of_option',
            'number_of_list',
            'number_of_th',
            'number_of_tr',
            'number_of_href',
            'number_of_paragraph',
            'number_of_script',
            'length_of_title',
            'has_h1',
            'has_h2',
            'has_h3',
            'length_of_text',
            'number_of_clickable_button',
            'number_of_a',
            'number_of_img',
            'number_of_div',
            'number_of_figure',
            'has_footer',
            'has_form',
            'has_text_area',
            'has_iframe',
            'has_text_input',
            'number_of_meta',
            'has_nav',
            'has_object',
            'has_picture',
            'number_of_sources',
            'number_of_span',
            'number_of_table',
            'URL'
        ]

        df = pd.DataFrame(data=data, columns=columns)
        df['label'] = 1

        # Check if the output file exists, and handle the header accordingly
        file_exists = os.path.exists(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)

        # Update state for the next run
        new_begin = end + 1
        new_end = end + 1000
        write_state(new_begin, new_end)
        print(f"Scraped URLs from {begin} to {end}. Next range is from {new_begin} to {new_end}.")

        # Sleep for a short period before the next iteration to avoid overloading
        time.sleep(10)


# Start the scraping process
scrape_and_save()
