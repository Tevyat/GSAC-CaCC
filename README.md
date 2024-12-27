# GSAC-CaCC
  ## ASK 
"The customer is always right, in matters of taste." This quote though popularized during the 1990s, is still passed around to this day, especially in industries where customer satisfaction can make or break a business. Within the aviation industry, the satisfaction of the passengers plays big role in shaping the reputation of an airline. This case study looks at these reviews to find patterns in what passenger’s value most and how airlines can improve their services. Specifically, this case study hopes to answer the following questions:

* What factors (Seat Comfort, Staff Service, Food & Beverages, etc.) have the most significant impact on the overall rating of the passenger?
* Do ratings vary significantly between people flying for leisure (solo, couple, or with family) and those flying for business?
* Which traveler type reports the highest satisfaction?
* Are there differences in satisfaction between airlines? Which airlines are rated highest or lowest?
  
&nbsp;  

## PREPARE
The dataset used was taken from Kaggle, named **“Airlines Review Dataset”** by user **“Elijah Alabi”**. It contains customer reviews for the top 10 rated airlines in 2023, including Singapore Airlines, Qatar Airways, ANA (All Nippon Airways), Emirates, Japan Airlines, Turkish Airlines, Air France, Cathay Pacific Airways, EVA Air, and Korean Air. It provides insights into passenger satisfaction and service quality aspects, ranging from seat comfort to inflight entertainment. The dataset consists of 8,100 reviews with 17 columns, including both numerical and categorical data. These are the columns:
- Title: Title of the review.
- Name: Name of the reviewer.
- Review Date: Date when the review was posted.
- Airline: Airline being reviewed.
- Verified: Whether the review is verified.
- Reviews: Text of the review.
- Type of Traveller: Type of traveler (e.g., Solo Leisure, Family Leisure).
- Month Flown: Month of the flight.
- Route: Route of the flight.
- Class: Class of travel (e.g., Economy Class, Business Class).
- Seat Comfort: Rating for seat comfort (1-5).
- Staff Service: Rating for staff service (1-5).
- Food & Beverages: Rating for food and beverages (1-5).
- Inflight Entertainment: Rating for inflight entertainment (1-5).
- Value For Money: Rating for value for money (1-5).
- Overall Rating: Overall rating for the flight (1-10).
- Recommended: Whether the reviewer recommends the airline (yes/no)

[Source](https://www.kaggle.com/datasets/elijahconnectng/airlines-review-dataset)

&nbsp;  

## PROCESS

These were the steps taken to process the dataset for analysis using the Python programming language (as can be seen on the [GDAC:CaCC Colab Notebook](https://colab.research.google.com/drive/1cbcP1dgyi5l9U1PpfIzZccnTZXrew3lK?usp=sharing)):

&nbsp;

```python
df = df[df['Verified'] == "True"]
```
This line filters the dataset to include only reviews that are marked as "verified." The column Verified contains values like True or False, indicating whether the reviewer was a verified customer.

&nbsp;

```python
df = df.drop(["Title", "Name", "Review Date", "Verified", "Month Flown", "Reviews", "Route", "Class", "Recommended"], axis=1)
```
These columns are considered irrelevant for the current analysis, so they are removed to streamline the dataset.

&nbsp;

```python
df = df[~df['Airline'].isin(['All Nippon Airways', 'Japan Airlines', 'Korean Air', 'EVA Air'])]
```
This line excludes rows where the airline is one of the following: All Nippon Airways, Japan Airlines, Korean Air, and EVA Air. This was done due to the airlines having less than 500 reviews.

&nbsp;

```python
df['Overall Rating'] = MinMaxScaler(feature_range=(1, 5)).fit_transform(df[['Overall Rating']])
```
This step normalizes the Overall Rating column, which has ratings between 1 and 10. The MinMaxScaler from the sklearn.preprocessing library scales the ratings to a new range of 1 to 5. This was done to match the scale of the other ratings.

&nbsp;

```python
print(df.isnull().sum())
```
Lastly, this line checks the sum of null values from each column. This returned a total of 0 null values for each column.

&nbsp;

# ANALYZE

### Descriptive Statistics

This showcases the descriptive statistics for the categorical data within the dataset.

```python
Type of Traveller       Business  Couple Leisure  Family Leisure  Solo Leisure
Airline                                                                       
Air France                   123             179              72           240
Cathay Pacific Airways        88             129              97           251
Emirates                     178             243             216           360
Qatar Airways                228             288             204           591
Singapore Airlines           113             209             150           279
Turkish Airlines             244             256             280           489
```

This showcases the descriptive statistics for the numerical data within the dataset.
```python
Singapore Airlines:
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count    751.000000     751.000000        751.000000              751.000000   
mean       3.668442       3.902796          3.536618                3.882823   
std        1.307640       1.399002          1.426765                1.172853   
min        1.000000       1.000000          1.000000                1.000000   
25%        3.000000       3.000000          2.000000                3.000000   
50%        4.000000       5.000000          4.000000                4.000000   
75%        5.000000       5.000000          5.000000                5.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count       751.000000      751.000000  
mean          3.452730        6.549933  
std           1.509115        3.276966  
min           1.000000        1.000000  
25%           2.000000        4.000000  
50%           4.000000        8.000000  
75%           5.000000        9.000000  
max           5.000000       10.000000  

Qatar Airways:
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count   1311.000000    1311.000000       1311.000000             1311.000000   
mean       3.955759       4.291381          3.932876                4.130435   
std        1.241593       1.140742          1.258283                1.023650   
min        1.000000       1.000000          1.000000                1.000000   
25%        3.000000       4.000000          3.000000                4.000000   
50%        4.000000       5.000000          4.000000                4.000000   
75%        5.000000       5.000000          5.000000                5.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count      1311.000000     1311.000000  
mean          3.711670        7.072464  
std           1.394376        3.171692  
min           1.000000        1.000000  
25%           3.000000        5.000000  
50%           4.000000        8.000000  
75%           5.000000       10.000000  
max           5.000000       10.000000  

UAE (Emirates):
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count    997.000000     997.000000        997.000000              997.000000   
mean       3.128385       2.939819          2.950853                3.670010   
std        1.396196       1.595170          1.463955                1.355649   
min        1.000000       1.000000          1.000000                1.000000   
25%        2.000000       1.000000          2.000000                3.000000   
50%        3.000000       3.000000          3.000000                4.000000   
75%        4.000000       5.000000          4.000000                5.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count       997.000000      997.000000  
mean          2.674022        4.520562  
std           1.506031        3.287867  
min           1.000000        1.000000  
25%           1.000000        1.000000  
50%           2.000000        4.000000  
75%           4.000000        8.000000  
max           5.000000       10.000000  

Turkish Airlines:
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count   1269.000000    1269.000000       1269.000000             1269.000000   
mean       2.747045       2.879433          2.972419                3.092199   
std        1.361886       1.523360          1.476962                1.419838   
min        1.000000       1.000000          1.000000                1.000000   
25%        1.000000       1.000000          2.000000                2.000000   
50%        3.000000       3.000000          3.000000                3.000000   
75%        4.000000       4.000000          4.000000                4.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count      1269.000000     1269.000000  
mean          2.375099        3.620961  
std           1.554847        3.240210  
min           1.000000        1.000000  
25%           1.000000        1.000000  
50%           2.000000        2.000000  
75%           4.000000        7.000000  
max           5.000000       10.000000  

Air France:
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count    614.000000     614.000000        614.000000              614.000000   
mean       2.920195       3.298046          3.094463                3.174267   
std        1.370327       1.558253          1.465493                1.372269   
min        1.000000       1.000000          1.000000                1.000000   
25%        2.000000       2.000000          2.000000                2.000000   
50%        3.000000       4.000000          3.000000                3.000000   
75%        4.000000       5.000000          4.000000                4.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count       614.000000      614.000000  
mean          2.672638        4.664495  
std           1.617200        3.539773  
min           1.000000        1.000000  
25%           1.000000        1.000000  
50%           2.000000        3.000000  
75%           4.000000        8.000000  
max           5.000000       10.000000  
\Cathay Pacific Airways:
       Seat Comfort  Staff Service  Food & Beverages  Inflight Entertainment  \
count    565.000000     565.000000        565.000000              565.000000   
mean       3.598230       3.596460          3.175221                3.789381   
std        1.313999       1.464711          1.440088                1.199835   
min        1.000000       1.000000          1.000000                1.000000   
25%        3.000000       2.000000          2.000000                3.000000   
50%        4.000000       4.000000          3.000000                4.000000   
75%        5.000000       5.000000          4.000000                5.000000   
max        5.000000       5.000000          5.000000                5.000000   

       Value For Money  Overall Rating  
count       565.000000      565.000000  
mean          3.320354        6.099115  
std           1.489333        3.243775  
min           1.000000        1.000000  
25%           2.000000        3.000000  
50%           4.000000        7.000000  
75%           5.000000        9.000000  
max           5.000000       10.000000
```
&nbsp;

To answer the research questions from the beginning, we will be using data visualization with the Pandas module from the Python programming language.

* What factors (Seat Comfort, Staff Service, Food & Beverages, etc.) have the most significant impact on the overall rating of the passenger?
```python
correlation_matrix = df[['Seat Comfort', 'Staff Service', 'Food & Beverages', 
                         'Inflight Entertainment', 'Value For Money', 'Overall Rating']].corr()
                         
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix with Normalized Overall Rating')
plt.show()
```
<p align=center><img src=https://github.com/user-attachments/assets/2f5f7662-829d-4136-aba9-7b36fcadac03><br>
Figure 4.1 </p>
As can be seen from this heatmap, 
* Value for Money has the strongest correlation with the overall rating (0.89).
* Staff Service also shows a moderate positive correlation (0.21).
* Other factors, including Seat Comfort, Food & Beverages, and Inflight Entertainment, have weaker correlations with the overall rating (0.19, 0.16, and 0.15 respectively).
This suggests that "Value for Money" significantly influences the overall rating compared to other factors.

&nbsp;

* Do ratings vary significantly between people flying for leisure (solo, couple, or with family) and those flying for business?
* Which traveler type reports the highest satisfaction?
* Are there differences in satisfaction between airlines? Which airlines are rated highest or lowest?
