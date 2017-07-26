from PyLyrics import *
import time
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd


def getLyrics(artist,song):

    try:
        l = PyLyrics.getLyrics(artist, song)
    except:
        return ''

    return l


x = pd.read_csv('songs_to_scrape.csv',header=0)

print x.shape[0]

for i in range(300000,400000):

    artist = x.iloc[i,1]

    song = x.iloc[i,2]

    lyric = getLyrics(artist,song)

    x.iloc[i,3] = lyric

    if i % 1000 ==0:

        print i , pd.notnull(x['lyrics']).sum()
        x.to_csv('songs_to_scrape.csv',header=True,index=None)


