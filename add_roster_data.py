from sportsipy.ncaab.teams import Team, Roster, Teams
import pickle
import json
from threading import Thread
from queue import Queue

def add_roster_data(data):
    """Takes a set of data team data and adds roster features as well"""

    # Add each row into a queue
    in_queue = Queue()
    for i in range(0,len(data)):
        in_queue.put(data.iloc[i])

    # Create threads
    thread_count = 20
    threads = []
    for t in range(0, thread_count):
        threads.append(Thread(target=get_yearly_roster))