import requests
import json
import time
import os

def smart_request(url):
    """Since sports reference now has a limit of 20 requests per minute, this
    function is designed to make sure you never go over that threshold"""

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
    response = requests.get(url)

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
        response = smart_request("https://www.sports-reference.com/cbb/seasons/men/2022-advanced-school-stats.html")
        print(response)
        print("Done!")