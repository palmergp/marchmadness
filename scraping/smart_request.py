import requests
import json
import time
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

options = Options()
options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"  # Update this path


def selenium_request(url):
    """Makes the request using selenium instead of the requests library.
    This gets around a lot of anti-scraping checks
    """
    #driver_path = "C:\\Webdriver\\chromedriver.exe"
    driver_path = "C:\\Webdriver\\geckodriver.exe"

    # Create a new instance of the Chrome driver
    service = Service(driver_path)
    driver = webdriver.Firefox(service=service, options=options)

    # Make the request
    # This will open an instance of Chrome
    driver.get(url)

    # Grab the HTML
    html_content = driver.page_source

    # Close the driver
    driver.close()

    return html_content


def smart_request(url, selenium=True):
    """Since sports reference now has a limit of 20 requests per minute, this
    function is designed to make sure you never go over that threshold

    Inputs:
        - url: (str) URL that should be pulled
        - selenium: (bool) flag indicating whether selenium should be used. If false, requests will be used
    """

    ### First load log of last requests
    # Get the directory name of this file
    dirname = os.path.dirname(__file__)
    # Get the full path of the adjacent file
    filename = os.path.join(dirname, 'recent_requests.json')
    # Open file
    try:
        f = open(filename, "r")
        recent_requests = json.load(f)
        f.close()
    except FileNotFoundError:
        recent_requests = []

    # Remove any timestamps that are more than a minute old
    recent_requests = [x for x in recent_requests if time.time() - x <= 60]
    if len(recent_requests) >= 15:
        # If there have been 15 requests, wait to avoid being timed out
        sleep_time = 60 - (time.time() - recent_requests[0]) + 1  # Add one second to be safe
        print(f"Request cap met. Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

    # Make the request
    if selenium:
        response = selenium_request(url)
    else:
        response = requests.get(url)
        response = response.text

    # Log the time
    request_time = time.time()
    recent_requests.append(request_time)

    # Update the json to have
    f = open(filename, 'w')
    json.dump(recent_requests, f)
    f.close()

    # Return the response
    return response


if __name__ == "__main__":
    while True:
        response_text = smart_request("https://www.sports-reference.com/cbb/seasons/men/2022-advanced-school-stats.html")
        print(response_text)
        print("Done!")
