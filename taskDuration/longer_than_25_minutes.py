import re
import numpy as np
import pandas as pd
#from zk_stats import *

relevant_columns = ['neutral_writing', 'lemon_writing', 'fish_writing']

df = pd.read_csv("all_task_durations.csv")

min_to_ms = 25*60000

neutral_long_sessions = []
lemon_long_sessions = []
fish_long_sessions = []

for col in relevant_columns:
    greater_than_idx = np.where(df[col] > min_to_ms)[0]
    num_greater_than = len(greater_than_idx)
    greater_than_list = list(df.loc[greater_than_idx, col])
    
    # getting difference with 25 minutes and converting back to minutes
    amount_over_25_minutes = [(el - min_to_ms)/60000 for el in greater_than_list]

    if re.search("neutral", col):
        neutral_long_sessions = amount_over_25_minutes
    elif re.search("lemon", col):
        lemon_long_sessions = amount_over_25_minutes
    elif re.search("fish", col):
        fish_long_sessions = amount_over_25_minutes

print(neutral_long_sessions)
print(lemon_long_sessions)
print(fish_long_sessions)



