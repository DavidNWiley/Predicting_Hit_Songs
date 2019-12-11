#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:15:09 2019

@author: david
"""

import numpy as np, pandas as pd
import seaborn as sns

songs = pd.read_csv("all_song_data.csv")

songs.head()

# matplotlib lets us create the histogram plots to visualize the data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# creating an array of the headers in the data to cycle through a for loop to plot each histogram
characteristics = ['energy', 
                   'liveness', 
                   'tempo', 
                   'speechiness', 
                   'acousticness', 
                   'instrumentalness', 
                   'time_signature', 
                   'danceability', 
                   'key', 
                   'duration_ms', 
                   'loudness', 
                   'valence', 
                   'mode']


# creating a dataframe specifically for the billboard hits to be compared visually in the histograms
popular_songs = songs.loc[songs['popularity']==1]

unpopular_songs = songs.loc[songs['popularity']==0]

# creating an empty plot grid
grid = plt.figure()
# the count variable will be used to add each histogram to the grid
count = 1

# a for loop to create each histogram
for name in characteristics:
    
    # Capitalizing the headers in the name array to be used as titles
    title = name.capitalize()
    
    # adding the histogram for the iteration to the plot grid
    # there are two histograms being generated (one with all data, one with just hit songs)
    # they are layered ("stacked") for comparison
    grid.add_subplot(7,2,count)
    plt.hist(songs[name], color='blue', label='All Songs', alpha=0.7)
    plt.hist(popular_songs[name], color='red', label='Popular Songs', alpha=0.7)
    plt.hist(unpopular_songs[name], color='navy', label='Unpopular Songs', histtype='step')
    plt.legend(prop={'size': 10})
    
    # applying title and axis names to the histogram
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    # adding 1 to the count to position the next histogram plot
    count+=1

# configuring the size of plot grid
grid.set_figheight(50)
grid.set_figwidth(15)

# showing plot grid
plt.show()

attributes = ['popularity', 
              'energy', 
              'liveness', 
              'tempo', 
              'speechiness', 
              'acousticness', 
              'instrumentalness', 
              'time_signature', 
              'danceability', 
              'key', 
              'duration_ms', 
              'loudness', 
              'valence', 
              'mode']

corr_matrix = songs[attributes].corr()

corr_matrix["popularity"].sort_values(ascending=False)



main_att = ['popularity', 
              'energy',   
              'speechiness', 
              'acousticness', 
              'instrumentalness', 
              'time_signature', 
              'danceability',  
              'loudness', 
              'valence']

plt.figure(figsize=(8, 8))
sns.heatmap(songs[main_att].corr(), annot=True, linewidths = .5, cmap='RdBu_r')

from pandas.plotting import scatter_matrix


songs_df = pd.DataFrame(songs, columns=main_att)
#pd.plotting.scatter_matrix(songs_df, alpha=0.2, figsize=(20, 20))
#plt.show()

sns.pairplot(songs_df)
plt.figure(figsize=(8, 8))
plt.show()


for name in characteristics:
    
    # Capitalizing the headers in the name array to be used as titles
    title = name.capitalize()

    plt.hist(songs[name], color='blue', label='All Songs', alpha=0.7)
    plt.hist(popular_songs[name], color='red', label='Popular Songs', alpha=0.7)
    plt.hist(unpopular_songs[name], color='navy', label='Unpopular Songs', histtype='step')
    plt.legend(prop={'size': 10})
    
    # applying title and axis names to the histogram
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    # adding 1 to the count to position the next histogram plot
    plt.show()
# configuring the size of plot grid
grid.set_figheight(50)
grid.set_figwidth(15)

