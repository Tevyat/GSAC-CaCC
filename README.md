# GSAC-CaCC

This case study was done as part of a requirement for the Google Data Analytics Professional Certificate, Course 8: Google Data Analytics Capstone: Complete a Case Study.

&nbsp;  

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
df = df.drop(["Title", "Name", "Review Date", "Verified", "Reviews", "Route", "Class", "Recommended"], axis=1)
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

