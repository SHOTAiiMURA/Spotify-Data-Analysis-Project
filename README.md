# Spotify-Data-Analysis-Project

 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
tracks_data = pd.read_csv("tracks_data.csv")
feature_data = pd.read_csv("SpotifyFeatures.csv")
```
```python
tracks_data.head()
```
<img width="742" alt="Screenshot 2024-09-22 at 22 23 44" src="https://github.com/user-attachments/assets/ba206a28-02a6-4af8-81a7-dae5350c40ed">

```python
#checking null in tracks data
pd.isnull(tracks_data).sum()
```
<img width="192" alt="Screenshot 2024-09-22 at 22 24 03" src="https://github.com/user-attachments/assets/6a620bba-bee4-4472-ad75-78663509deae">

```python
#checking null in feature data
pd.isnull(feature_data).sum()
```
<img width="183" alt="Screenshot 2024-09-22 at 22 24 44" src="https://github.com/user-attachments/assets/34958683-e40d-447d-85a1-e136180ae3af">

```python
#Checking information in tracks data
tracks_data.info()
```
<img width="386" alt="Screenshot 2024-09-22 at 22 25 08" src="https://github.com/user-attachments/assets/f8fe4760-73d3-46d5-8c23-184a117c7d03">

```python
#checking information in feature data
feature_data.info()
```
<img width="389" alt="Screenshot 2024-09-22 at 22 25 44" src="https://github.com/user-attachments/assets/21f35080-5c80-49b5-a128-448cc7ca7030">

```python
#Reaserching the 10 least popular songs in the Spotify
leastSongs = tracks_data.sort_values("popularity",ascending=True)[0:10]
leastSongs[["name","popularity"]]
```
<img width="472" alt="Screenshot 2024-09-22 at 22 26 06" src="https://github.com/user-attachments/assets/a6ab35ae-8923-463c-98db-50b24d8c5eca">

```python
s = "Python syntax highlighting"
print s
```
```python
#descriptive statistics of tracks
tracks_data.describe().transpose()
```
<img width="741" alt="Screenshot 2024-09-22 at 22 26 34" src="https://github.com/user-attachments/assets/6a3aa3a4-aa6f-48e9-ac23-c6510cd04ad7">

```python
#descriptive of feature
feature_data.describe().transpose()
```
<img width="743" alt="Screenshot 2024-09-22 at 22 26 56" src="https://github.com/user-attachments/assets/47e1f185-5a86-442f-9245-56ea7747a117">

```python
leastSongs=tracks_data
popularSongs = leastSongs[leastSongs["popularity"]>90].sort_values("popularity",ascending=False)[:10]
popularSongs[["name","popularity","artists"]]
```
<img width="672" alt="Screenshot 2024-09-22 at 22 27 14" src="https://github.com/user-attachments/assets/06962fce-392b-4ad2-b4e3-f74d4870f22d">

```python
tracks_data["duration"]=tracks_data["duration_ms"].apply(lambda x: round(x/1000))
tracks_data.drop('duration_ms', inplace = True, axis=1)
tracks_data.duration.head()
```
<img width="234" alt="Screenshot 2024-09-22 at 22 27 46" src="https://github.com/user-attachments/assets/a90d1ec0-adef-482a-8e33-14dcce12e9a6">

```python
tracks_data["duration"]=tracks_data["duration_ms"].apply(lambda x: round(x/1000))
tracks_data.drop('duration_ms', inplace = True, axis=1)
tracks_data.duration.head()
```
<img width="805" alt="Screenshot 2024-09-22 at 22 29 32" src="https://github.com/user-attachments/assets/f27f6bb8-6c8c-40cc-8113-eb34c07beaec">

```python
sample_sp=tracks_data.sample(int(0.004*len(tracks_data)))
print(len(sample_sp))
```
```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='loudness', x='energy', color='#ffac33').set(title='Regression Plot - Loudness vs Energy Correlation')
```
<img width="711" alt="Screenshot 2024-09-22 at 22 30 10" src="https://github.com/user-attachments/assets/13a622cc-d726-426b-b6b4-2bbf96bc92e3">

```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='energy', color='#3372ff').set(title='Regression Plot - Popularity vs Energy Correlation')
```
<img width="701" alt="Screenshot 2024-09-22 at 22 30 34" src="https://github.com/user-attachments/assets/488beb5e-0ff4-43f5-a0f7-df41f235a899">

```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='acousticness', color='#3372ff').set(title='Regression Plot - Popularity vs Acousticness Correlation')
```
<img width="686" alt="Screenshot 2024-09-22 at 22 31 01" src="https://github.com/user-attachments/assets/abe4fd11-9746-4a80-b91e-ae119d97f6d2">

```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='danceability', color='#3372ff').set(title='Regression Plot - Popularity vs Danceability Correlation')
```
<img width="678" alt="Screenshot 2024-09-22 at 22 31 29" src="https://github.com/user-attachments/assets/6bca1993-2d91-4d9c-abb0-08716fd87597">

```python
tracks_data['dates']=tracks_data.index.get_level_values('release_date')
tracks_data.dates=pd.to_datetime(tracks_data.dates,format='mixed')
years=tracks_data.dates.dt.year
tracks_data.head()
```
<img width="739" alt="Screenshot 2024-09-22 at 22 32 01" src="https://github.com/user-attachments/assets/e411eaca-972e-4537-a3e2-55bfe6bc2b54">

```python
sns.displot(years, discrete=True, aspect=2, height=4, kind='hist',color='g').set(title='Number of songs - per year')
```
<img width="732" alt="Screenshot 2024-09-22 at 22 32 28" src="https://github.com/user-attachments/assets/e4be9c09-ea8e-4a79-972c-89e108c80b81">

```python
total_dr = tracks_data.duration
fig_dims = (15,5)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title='Years vs Duration')
plt.xticks(rotation=90)
```
<img width="736" alt="Screenshot 2024-09-22 at 22 33 03" src="https://github.com/user-attachments/assets/0dfee50a-1c4f-4427-b11f-f2e175f40ad0">

```python
sns.set_style(style="whitegrid")
fig_dims=(10,5)
fig,ax=plt.subplots(figsize=fig_dims)
fig=sns.lineplot(x=years,y=total_dr,ax=ax).set(title="year vs duration")
plt.xticks(rotation=60)
```

```python
plt.title('Duration of songs in different Genres')
sns.color_palette('crest', as_cmap=True)
sns.barplot(y='genre', x='duration_ms', color="g", data=feature_data)
plt.xlabel('Duration in ms')
plt.ylabel('Genres')
```
<img width="661" alt="Screenshot 2024-09-22 at 22 34 31" src="https://github.com/user-attachments/assets/07e72d4f-8f8e-4f5f-8896-cb1a63f5eef3">

```python
sns.set_style(style='darkgrid')
plt.figure(figsize=(8,4))
Top = feature_data.sort_values('popularity', ascending=False)[:10]
sns.barplot(y = 'genre', x = 'popularity', data = Top).set(title='Genres by Popularity-Top 5')
```
<img width="732" alt="Screenshot 2024-09-22 at 22 34 41" src="https://github.com/user-attachments/assets/8a328e7e-d7ac-43f0-a843-82368d4631e0">

