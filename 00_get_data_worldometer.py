# 1. https://www.reddit.com/r/webscraping/comments/s190f2/web_scraping_from_httpswebarchiveorg_wayback/
# 2. https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server

import requests
from datetime import date, timedelta
import time
import os
import pandas as pd
from bs4 import BeautifulSoup
from random import randrange

def return_middle_timestamp(df):
    df = df.loc[(df['mimetype'] == "text/html")]
    # middle_index = len(df) // 2
    # middle_row = df.iloc[middle_index]
    # timestamp = middle_row['timestamp']

    random_index = randrange(len(df))
    random_row = df.iloc[random_index]
    timestamp = random_row['timestamp']
    return timestamp

start_date = date(2020, 12, 1)
end_date = date(2021, 4, 1)
delta = timedelta(days=1)
max_file_date = start_date

duration = 1  # seconds
freq = 440  # Hz

for data_file in os.listdir(os.path.join("data_cache", "worldometer")):
    if data_file.endswith(".csv"):
        file_date = date(int(data_file[:4]), int(data_file[4:6]), int(data_file[6:8]))
        # Update max_file_date if the current file_date is greater
        if file_date > max_file_date:
            max_file_date = file_date

start_date = max_file_date + delta
print("Updated start_date:", start_date)
headers = {"User-Agent": "'User-agent': 'Mozilla/5.0'",}

while start_date <= end_date:
    date_str = start_date.strftime("%Y%m%d")
    print("doing for: ", date_str)
    internet_archive_url = "https://web.archive.org/cdx/search/cdx?url=https://www.worldometers.info/coronavirus/&matchType=prefix&output=json&from=" + date_str + "&to=" + date_str
    attempts = 0
    while attempts < 3:
        try:
            r = requests.get(internet_archive_url, headers=headers)
            internet_archive_content = eval(r.content.decode("utf-8"))
            internet_archive_df = pd.DataFrame(internet_archive_content[1:], columns=internet_archive_content[0])
            timestamp = return_middle_timestamp(internet_archive_df)
            print("found timestamp: ", timestamp)
            worldometers_url = "https://web.archive.org/web/"+ timestamp + "/https://www.worldometers.info/coronavirus/"
            attempts_2 = 0
            while attempts_2 < 3:
                try:
                    r = requests.get(worldometers_url, headers=headers)
                    soup = BeautifulSoup(r.content, 'html5lib')
                    # below the date: date(2020, 3, 19)
                    # this works: table = soup.find_all("table", id="main_table_countries")
                    # above the date: date(2020, 3, 19)
                    # because: they changed the table id
                    table = soup.find_all("table", id="main_table_countries_today")
                    print(len(table), type(table), worldometers_url)
                    # before the date: date(2020, 3, 29)
                    # this works: worldometer_df = pd.read_html(str(table[0]))
                    # above the date: date(2020, 3, 29)
                    # because: https://stackoverflow.com/questions/62302639/pandas-no-tables-found-matching-pattern/62302842#62302842
                    worldometer_df = pd.read_html(str(table[0]), displayed_only=False)
                    path = os.path.join("data_cache", "worldometer", date_str + ".csv")
                    worldometer_df[0].to_csv(path, index=False)
                    print("saving: ", path)
                    break
                except Exception as e:
                    attempts_2 += 1
                    print(f"Attempt_2 {attempts_2} failed with error: {e}")
                    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                    time.sleep(30)
            start_date += delta
            break
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed with error: {e}")
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            time.sleep(30)