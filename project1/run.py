import os
import pandas as pd
import numpy as np
import deliverable1_alt as d1
import crawl as cr 


seeds = r'urls_from_tranco.csv'

seed_list = pd.read_csv(seeds)['url'].tolist()  


score_list = []
for url in seed_list:
    score = d1.score_url(url)
    print(score['score'])
    score_list.append(score['score'])

