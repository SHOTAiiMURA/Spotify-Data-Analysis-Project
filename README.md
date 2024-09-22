# Spotify-Data-Analysis-Project
![image](https://github.com/user-attachments/assets/3fc67c7d-703a-46b8-8a95-f0acb49b84ac)

# Introduction
# Requirement
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
# Exploring the codes and dataset
```python
#reading datasets
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

# If there is Null Values in the Dataset
```python
#checking null in feature data
pd.isnull(feature_data).sum()
```
<img width="183" alt="Screenshot 2024-09-22 at 22 24 44" src="https://github.com/user-attachments/assets/34958683-e40d-447d-85a1-e136180ae3af">

# Overview of dataset
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

# Analysing the dataset for insights
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

## Heatmap
```python
td = tracks_data.drop(['key','mode','explicit'], axis=1).corr(method = 'pearson')
plt.figure(figsize=(9,5))
hmap = sns.heatmap(td, annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, cmap='Greens', linewidths=0.1, linecolor='black')
hmap.set_title('Correlation HeatMap')
hmap.set_xticklabels(hmap.get_xticklabels(), rotation=90)
```
<img width="805" alt="Screenshot 2024-09-22 at 22 29 32" src="https://github.com/user-attachments/assets/f27f6bb8-6c8c-40cc-8113-eb34c07beaec">

**How to Read the Heatmap**
- Each cell shows the correlation between two features.
- The color of the cell represents the magnitude and direction of correlation coeffient.
- Green color represents positive correlation.
- Blue color represents negative correlation.
- Darker color indicates a stronger correlation.

**Interpretation**
- **Danceability and Energy:** As indicated by the bright green color, there is a strong positive correlation. In other words, songs with high danceability tend to also have high energy.
- **Acousticness and Loudness:** As indicated by the blue color, there is a strong negative correlation. In other words, songs with high acousticness (such as those using acoustic guitars) tend to have low loudness.

```python
sample_sp=tracks_data.sample(int(0.004*len(tracks_data)))
print(len(sample_sp))
```
<img width="48" alt="Screenshot 2024-09-22 at 23 08 36" src="https://github.com/user-attachments/assets/c724ce7c-1c0f-4af8-8031-34704e43666b">

##Regression Plot
```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='loudness', x='energy', color='#ffac33').set(title='Regression Plot - Loudness vs Energy Correlation')
```
<img width="711" alt="Screenshot 2024-09-22 at 22 30 10" src="https://github.com/user-attachments/assets/13a622cc-d726-426b-b6b4-2bbf96bc92e3">

**Explanation of the Regression Plot**

- This graph is called a regression plot and visually represents the relationship between two variables: loudness and energy of the songs.

**How to Read the Graph**

- X-axis: Energy (a measure of how lively or energetic a song is)
- Y-axis: Loudness (a measure of the volume of the song)
- Orange Dots: Represent individual song data, plotted according to their values on the X and Y axes.
- Orange Line: Represents the best-fitting line (regression line) for the data, indicating the strength and direction of the relationship between the two variables.
**Insights from the Graph**
**Relationship between Energy and Loudness:**
- The higher the energy of a song, the higher its loudness tends to be. This suggests that, in general, more energetic songs are also louder.

**Strength of the Relationship:**
- The slope of the regression line is positive, indicating a positive correlation between the two variables. However, there is significant variation in the data, meaning not all songs follow this trend strictly.

```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='energy', color='#3372ff').set(title='Regression Plot - Popularity vs Energy Correlation')
```
<img width="701" alt="Screenshot 2024-09-22 at 22 30 34" src="https://github.com/user-attachments/assets/488beb5e-0ff4-43f5-a0f7-df41f235a899">

**Explanation of the Regression Plot**

- This regression plot shows the relationship between two variables: popularity and energy of the songs.

**How to Read the Graph**

- X-axis: Energy (a measure of how lively or energetic a song is)
- Y-axis: Popularity (a measure of the song’s popularity)
- Blue Dots: Represent individual song data, plotted according to their values on the X and Y axes.
- Blue Line: Represents the best-fitting line (regression line) for the data, indicating the strength and direction of the relationship between the two variables.
**Insights from the Graph**

**Relationship between Energy and Popularity:**
- Songs with higher energy tend to have higher popularity. This suggests that, in general, more energetic songs are more likely to become popular.

**Strength of the Relationship:**
- The slope of the regression line is positive, indicating a positive correlation between the two variables. However, there is significant variation in the data, meaning not all songs follow this trend strictly.


```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='acousticness', color='#3372ff').set(title='Regression Plot - Popularity vs Acousticness Correlation')
```
<img width="686" alt="Screenshot 2024-09-22 at 22 31 01" src="https://github.com/user-attachments/assets/abe4fd11-9746-4a80-b91e-ae119d97f6d2">

**Explanation of the Regression Plot**

- This regression plot shows the relationship between two variables: popularity and acousticness of the songs.

**How to Read the Graph**
- X-axis: Acousticness (a measure of how much a song incorporates acoustic instruments)
- Y-axis: Popularity (a measure of the song’s popularity)
- Blue Dots: Represent individual song data, plotted according to their values on the X and Y axes.
- Blue Line: Represents the best-fitting line (regression line) for the data, indicating the strength and direction of the relationship between the two variables.
**Insights from the Graph**

**Relationship between Acousticness and Popularity:**
- Songs with higher acousticness (e.g., those using acoustic guitars) tend to have lower popularity. This suggests that, in general, acoustic songs are less likely to become popular.

**Strength of the Relationship:**
- The slope of the regression line is negative, indicating a negative correlation between the two variables. However, there is significant variation in the data, meaning not all songs follow this trend strictly.
```python
plt.figure(figsize=(8,4))
sns.regplot(data=sample_sp, y='popularity', x='danceability', color='#3372ff').set(title='Regression Plot - Popularity vs Danceability Correlation')
```
<img width="678" alt="Screenshot 2024-09-22 at 22 31 29" src="https://github.com/user-attachments/assets/6bca1993-2d91-4d9c-abb0-08716fd87597">

**Explanation of the Regression Plot**

- This regression plot shows the relationship between two variables: popularity and danceability (how easy it is to dance to a song).

**How to Read the Graph**

- X-axis: Danceability (a measure of how easy it is to dance to a song)
- Y-axis: Popularity (a measure of the song’s popularity)
- Blue Dots: Represent individual song data, plotted according to their values on the X and Y axes.
- Blue Line: Represents the best-fitting line (regression line) for the data, indicating the strength and direction of the relationship between the two variables.
**Insights from the Graph**

**Relationship between Danceability and Popularity:**
- Songs with higher danceability tend to have higher popularity. This suggests that, in general, songs that are easier to dance to are more likely to become popular.

**Strength of the Relationship:**
- The slope of the regression line is positive, indicating a positive correlation between the two variables. However, there is significant variation in the data, meaning not all songs follow this trend strictly.

## Based on those insights from this graph, the following can be considered:

- Music Production: When creating energetic, loudness and danceability songs, there is a tendency for these songs to become more popular.
- Music Streaming Services: By utilizing users' listening history and song characteristics, more personalized song recommendations can be provided.

```python
tracks_data['dates']=tracks_data.index.get_level_values('release_date')
tracks_data.dates=pd.to_datetime(tracks_data.dates,format='mixed')
years=tracks_data.dates.dt.year
tracks_data.head()
```
<img width="739" alt="Screenshot 2024-09-22 at 22 32 01" src="https://github.com/user-attachments/assets/e411eaca-972e-4537-a3e2-55bfe6bc2b54">

##Histogram Analysis
```python
sns.displot(years, discrete=True, aspect=2, height=4, kind='hist',color='g').set(title='Number of songs - per year')
```
<img width="732" alt="Screenshot 2024-09-22 at 22 32 28" src="https://github.com/user-attachments/assets/e4be9c09-ea8e-4a79-972c-89e108c80b81">

**Overview of the Graph**

- This graph is a type of chart called a histogram, and it shows the trend in the number of songs released each year over a certain period.

- X-axis: Year
- Y-axis: Number of songs released each year
**Insights from the Graph**

**Increase in the Number of Songs:**
- Looking at the entire graph, it is clear that the number of songs has been increasing year by year. In particular, there is a sharp increase after the 1980s. This can be attributed to factors such as the digitalization of the music industry and the spread of music streaming services.

**Fluctuation in Song Numbers:**
- The graph shows significant fluctuations in the number of songs released each year. These fluctuations could be influenced by trends in the music industry or social events. For example, a popular music genre in a particular year or changes in the economic climate may have impacted the increase or decrease in the number of songs.

**Key Insights from the Graph**

**Growth of the Music Industry:**
- The graph indicates that the music industry has been expanding year by year.

**Changes in Music Trends:**
- From the yearly fluctuations in the number of songs, it is possible to understand how music trends have evolved over time.
**Impact of External Factors:**
- The graph allows us to analyze how social events and technological advancements have impacted the music industry.
```python
total_dr = tracks_data.duration
fig_dims = (15,5)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title='Years vs Duration')
plt.xticks(rotation=90)
```
<img width="736" alt="Screenshot 2024-09-22 at 22 33 03" src="https://github.com/user-attachments/assets/0dfee50a-1c4f-4427-b11f-f2e175f40ad0">

**Explanation of the Graph**

- This graph is a type of chart called a histogram, visually representing changes in the average length of songs released each year.

- X-axis: Year
- Y-axis: Song length (in seconds)
**Insights from the Graph**

**Song Length Trends:**
- Overall, the length of songs appears to be relatively stable. There are no significant increases or decreases, indicating that the song lengths remain within a certain range over time.

**Yearly Variations:**
- Some slight fluctuations can be observed in song lengths from year to year. In certain years, shorter songs tend to dominate, while in others, longer songs are more common, showing subtle differences in yearly trends.

**Key Insights from the Graph:**

- Standardization of Song Length in the Music Industry:
Many songs fall within a certain length range, suggesting that a general standard for song length has been established in the music industry.

**Changes in Music Trends:**
- Minor fluctuations in song lengths each year provide insights into how music trends have shifted over time. For example, shorter songs might have been more popular in a particular year, or longer songs were favored in another.

**Changes in Listening Environments:**
- The changes in song length may also be related to shifts in how people listen to music. For instance, the rise of smartphones might have led to the popularity of shorter songs that are easier to listen to on the go.

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

**Explanation of the Graph**

- This graph is a type of chart called a histogram, visually representing the average length of songs across different music genres.
- X-axis: Song length (in milliseconds)
- Y-axis: Music genres
**Insights from the Graph**

**Song Length by Genre:**
- The length of the bars for each genre represents the average song length in that genre.

**Genre Characteristics:**
- Longer songs: Genres like Classical, Opera, and Soundtrack tend to have longer average song lengths.
- Shorter songs: Genres such as Pop, Rock, and Hip-Hop tend to have shorter average song lengths.
**Diversity within Genres:**
- Even within the same genre, there is noticeable variation in song lengths.

**Key Insights from the Graph**
**Characteristics of Music Genres:**
- The tendencies in song length for each genre can provide insights into the genre's characteristics and historical background.

**Relationship with Listening Environment:**
- Song length is likely closely related to the environment in which the music is consumed and the preferences of the listeners.
```python
sns.set_style(style='darkgrid')
plt.figure(figsize=(8,4))
Top = feature_data.sort_values('popularity', ascending=False)[:10]
sns.barplot(y = 'genre', x = 'popularity', data = Top).set(title='Genres by Popularity-Top 5')
```
<img width="732" alt="Screenshot 2024-09-22 at 22 34 41" src="https://github.com/user-attachments/assets/8a328e7e-d7ac-43f0-a843-82368d4631e0">

**Explanation of the Graph**
- This graph is a type of chart called a bar chart, visually representing the top five most popular music genres and their popularity.
- X-axis: Popularity (higher values indicate greater popularity)
- Y-axis: Music genres

**Insights from the Graph**

**Top Popular Genres:**
- From this graph, it is clear that the top five most popular genres are "Reggaeton," "Hip-Hop," "Rap," "Pop," and "Dance."

**Differences in Popularity:**
- There is a noticeable difference in popularity among the genres. The graph visually confirms that Reggaeton is the most popular genre, while Dance ranks fifth.

**Key Insights from the Graph:**

**Current Music Trends:**
- This graph indicates which music genres are currently the most popular in the music scene.

**Target Audience Analysis:**
- By analyzing the characteristics of each genre, one can infer the demographics and preferences of their listeners.

**Trends in the Music Market:**
- By analyzing the shifts in popularity among genres, insights can be gained regarding trends in the music market.

## Summary
### Applications of Histogram Analysis and Regression Plots in Business

**Music Streaming Services:**

**Personalization:**
- By utilizing user listening histories, song lengths, genres, and other data, more personalized song recommendations can be made.

**Playlist Creation:**
- Based on trends in popular genres and song lengths, engaging playlists can be created.

**Record Labels:**

- Song Production:
- Strategies can be developed for creating hit songs by referencing the characteristics of popular genres and song lengths.

**Artist Development:**
- New artists can be discovered, and song production and promotion can be tailored to maximize their strengths.

**Music Events:**

**Event Planning:**
- Songs and artists can be selected to match the interests of attendees, leading to more appealing events.

**Ticket Sales:**
- Based on past data, pricing and sales strategies for tickets can be developed.

**Marketing:**

**Target Audience Identification:**
- By linking song characteristics with listener demographics, more effective targeting can be achieved.

**Promotion Strategies:**
- When developing promotional strategies for songs, optimal methods can be selected based on song features and target audiences.

## Future Prospects

**Utilization of Deep Learning:**
- Research is advancing in music generation using deep learning and automatic classification of songs. Leveraging these technologies will enable more sophisticated music analysis.

**Real-Time Analysis:**
- Using real-time data from streaming services to rapidly understand music trends will be crucial for responsive business development.

**Integration with Diverse Data:**
- By combining music data with various types of data, such as social media and economic data, more comprehensive analyses can be conducted.
