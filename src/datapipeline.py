import numpy as np
import pandas as pd
import json

from tqdm import tqdm

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius as lg

# API access tokens
# NOTE: NEED TO BE FILLED IN PRIOR TO CODE EXECUTION
sp_cid = 'Insert cid here'
sp_secret = 'Insert secret here'
client_credentials_manager = SpotifyClientCredentials(client_id=sp_cid, client_secret=sp_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

genius=lg.Genius(access_token='Insert access_token', skip_non_songs=True, remove_section_headers=True,
                 sleep_time=0.1, verbose=False)

def genre_song_collector(songs_per_genre=1000):
  """
  Pulls songs from the Spotify API using Spotipy and
  returns a dataframe of songs consisting of these features:
  track_name, artist_name, album_name, genre, duration_ms,
  popularity, explicit, track_id, artist_id

  Parameters:
  -----------
  songs_per_genre (int): How many songs you would like per genre

  Output:
  -------
  Returns a dataframe object
  """
  # Gather list of genres to iterate through
  genre_lst = sp.recommendation_genre_seeds()['genres']

  # Empty lists for desired features
  track_name = []
  artist_name = []
  album_name = []
  genre = []
  duration_ms = []
  popularity = []
  explicit = []
  track_id = []
  artist_id = []

  for g in tqdm(genre_lst):
    # Requests are limited to 50 units, so we need multiple API requests to get more songs
    for i in range(0,songs_per_genre,50):
      # Query
      q = 'genre:'+str(g)
      # Store API request results in a variable for extraction
      genre_results = sp.search(q=q, type='track', limit=50,offset=i)
      # Iterate through tracks and store relevant information in lists
      for i, t in enumerate(genre_results['tracks']['items']):
          track_name.append(t['name'])
          artist_name.append(t['artists'][0]['name'])
          album_name.append(t['album']['name'])
          genre.append(g)
          duration_ms.append(t['duration_ms'])
          popularity.append(t['popularity'])
          explicit.append(t['explicit'])
          track_id.append(t['id'])
          artist_id.append(t['artists'][0]['id'])

  return pd.DataFrame({'track_name':track_name,'artist_name':artist_name,
                       'album_name':album_name,'genre':genre,'duration_ms':duration_ms,
                       'popularity':popularity,'explicit':explicit,
                       'track_id' : track_id,'artist_id':artist_id})


def audio_feature_collector(track_id_lst):
  """
  Takes a list of track id's and returns a dataframe filled
  with the audio features for the provided songs.

  Parameters:
  -----------
  track_id_lst: list of track id's

  Output:
  -------
  Returns a dataframe object
  """
  audio_features = []
  batchsize = 100

  # Iterate over 100 song batches (due to API limit per request)
  for i in tqdm(range(0,len(track_id_lst),batchsize)):
    batch = track_id_lst[i:i+batchsize]
    # Collect features for 100 tracks
    feature_results = sp.audio_features(batch)
    # Store individual track info in list
    for track in feature_results:
      if track is not None:
        audio_features.append(track)

  df = pd.DataFrame.from_dict(data=audio_features,orient='columns')
  # Rename column to match column name from song dataframe for merge
  return df.rename(columns={'id':'track_id'})


def lyric_collector(track_lst,artist_lst):
  """
  Takes two ordered, same-size lists of tracks and artists and
  returns a list of each track's lyrics.

  Parameters:
  -----------
  track_lst: list of song titles
  artist_lst: list of artist names

  Output:
  -------
  Returns a list object
  """
  lyric_lst = []

  # Iterate through tracks and store lyrics
  for t, a in tqdm(zip(track_lst, artist_lst)):
    song = genius.search_song(title = t,
                              artist = a,
                              get_full_info = False)
    if song is not None:
      lyric_lst.append(song.lyrics)
    else:
      lyric_lst.append('')

  return lyric_lst


if __name__ == "__main__":
  pass