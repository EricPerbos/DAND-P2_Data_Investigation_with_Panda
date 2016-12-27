# Surviving the Titanic: "Ticket please !"  
 
_by Eric Perbos-Brinck in fulfillment of Udacity’s Data Analyst Nanodegree, Project 2_  
</br>

#### The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

#### One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

#### In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.

##### https://www.kaggle.com/c/titanic

### Project description:

#### In this project, we are going to explore the dataset in *.csv of 891 passengers of the Titanic which includes key information such as the variables below:

- Survival (0 = No; 1 = Yes)
- Pclass (Passenger Class: 1 = 1st; 2 = 2nd; 3 = 3rd); serves as a proxy for socio-economic status
- Name
- Sex
- Age
- Sib/Sp (number of Siblings/Spouses Aboard)
- Parch (number of Parents/Children Aboard)
- Ticket Number
- Passenger Fare
- Cabin Number
- Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

### Questions:

#### The main question we will try to answer: which single variable had the strongest impact on the survival rate of each passenger, if possible in a percentage/probability term ?

#### Then we will try to identify combinations of variables that may have an even stronger impact on the survival rate, beside the single variable above.


```python
# Imports
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
sns.set_style('whitegrid')
%matplotlib inline

# We upload the *.csv file
titanic_df = pd.read_csv('titanic_data.csv', header=0)
```

#### We start with a first look at the data with a serie of Pandas functions.


```python
titanic_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object




```python
titanic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

#### The titanic_df.info() indicates that three colums lack some data and may require some data wrangling, if possible.
- Age : 177 values missing (likely)
- Cabin : 687 values missing (unlikely)
- Embarked : 2 values missing (easy)


```python
titanic_df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



#### The titanic_df.describe() provides two immediate observations:
- Survival rate: only 38% of the 891 passengers survived
- Age: the average age of all passengers was 29.7 years with a strong StDev of 14.5 years

## Univariate Analysis: a first serie of visualizations of the passengers' profiles.


```python
# Writing a serie of sns.xxxplot() function to avoid repetitive code
def countplot(column):
    return sns.countplot(x= column, data= titanic_df)

def boxplot(column1, column2):
    return sns.boxplot(x= column1, y= column2, data= titanic_df)

def swarmplot(column1, column2):
    return sns.swarmplot(x= column1, y= column2, data= titanic_df)

def barplot(column1, column2):
    return sns.barplot(x= column1, y= column2, data= titanic_df, ci= None)
```
</br>

```python
countplot("Pclass")
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa1d8eb8>




![png](images/output_13_1.png)


#### There were more passengers travelling in 3rd Class than in 2nd and 1st Class -combined-. 
</br>

```python
countplot("Sex")
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa20ad30>




![png](images/output_15_1.png)


#### Male passengers outnumbered female passengers almost 2 to 1.
</br>

```python
countplot("Embarked")
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa35c630>




![png](images/output_17_1.png)


 S = Southampton; C = Cherbourg; Q = Queenstown
#### An overwhelming majority of the passengers embarked in Southampton, its first port for collecting passengers.
https://en.wikipedia.org/wiki/RMS_Titanic#Collecting_passengers


```python
# An histogram of ages, in 16 bins, NA values dropped
titanic_df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5).plot()
```




    []




![png](images/output_19_1.png)


#### This basic histogram shows the age repartition of 714 passengers (177 missing ) with a small long-tail in 55+

## Further Exploration with multiple-variable functions

#### Before wrangling the data, let's run a second set of exploratory functions.


```python
swarmplot('Pclass', 'Age')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xafeedd8>




![png](images/output_22_1.png)


#### The age of passengers in each class, with a swarmplot (aka. "beeswarm")
http://stanford.edu/~mwaskom/software/seaborn-dev/generated/seaborn.swarmplot.html
#### While the 1st Class has a rather homogeneous spread between 18 to 60, we can observe a clear overweight of the 18 to 35 range in 3rd Class


```python
boxplot('Pclass', 'Age')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb1dfd30>




![png](images/output_24_1.png)


#### The age of passengers in each class, with a boxplot showing the quartiles (box) and the outliers (dots)
https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.boxplot.html


```python
swarmplot('Sex', 'Age')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb669080>




![png](images/output_26_1.png)


#### The age of male and female passengers in swarmplot, with an overweight of 18 to 35 among male passengers.


```python
boxplot("Sex", "Age")
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb5d93c8>




![png](images/output_28_1.png)


#### The age of male and female passengers in boxplot

## Focused Exploration around the "Survived" variable.
#### We now start to explore the data to answer our main question:
- Which single variable had the strongest impact on the survival rate of each passenger, if possible in a percentage/probability term ?


```python
# Writing a *.groupby().mean() function to avoid repetitive code
def gb_mean(column):
    return titanic_df.groupby(column).mean()
```


```python
gb_mean('Survived')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>447.016393</td>
      <td>2.531876</td>
      <td>30.626179</td>
      <td>0.553734</td>
      <td>0.329690</td>
      <td>22.117887</td>
    </tr>
    <tr>
      <th>1</th>
      <td>444.368421</td>
      <td>1.950292</td>
      <td>28.343690</td>
      <td>0.473684</td>
      <td>0.464912</td>
      <td>48.395408</td>
    </tr>
  </tbody>
</table>
</div>



#### The titanic_df.groupby('Survived').mean() provides a first discrepancy between suvivors and non-survivors in the Pclass column, let's explore it further.


```python
gb_mean('Pclass')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
  </tbody>
</table>
</div>



#### Here we get the first hint: the lower the cabin class of the passenger (1st to 3rd), the lower its survival rate (63% to  24%).


```python
barplot('Pclass', 'Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa35ca20>




![png](images/output_36_1.png)


#### The same comparison of survival by class visualized in a barplot.

#### We continue the exploration with "Sex" column, female vs. male.


```python
gb_mean('Sex')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>431.028662</td>
      <td>0.742038</td>
      <td>2.159236</td>
      <td>27.915709</td>
      <td>0.694268</td>
      <td>0.649682</td>
      <td>44.479818</td>
    </tr>
    <tr>
      <th>male</th>
      <td>454.147314</td>
      <td>0.188908</td>
      <td>2.389948</td>
      <td>30.726645</td>
      <td>0.429809</td>
      <td>0.235702</td>
      <td>25.523893</td>
    </tr>
  </tbody>
</table>
</div>



#### Here we get the second hint: female passengers had a very signficantly higher rate of survival than male passengers (74% vs 19%)


```python
barplot('Sex', 'Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa4587f0>




![png](images/output_40_1.png)


#### The same comparison of survival by "Sex" visualized in a barplot.

#### We continue the exploration with "Embarked" column, indicating in which port the passenger embarked.
- C: Cherbourg (France)
- Q: Queenstown (UK)
- S: Southampton (UK)


```python
gb_mean('Embarked')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Embarked</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>445.357143</td>
      <td>0.553571</td>
      <td>1.886905</td>
      <td>30.814769</td>
      <td>0.386905</td>
      <td>0.363095</td>
      <td>59.954144</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>417.896104</td>
      <td>0.389610</td>
      <td>2.909091</td>
      <td>28.089286</td>
      <td>0.428571</td>
      <td>0.168831</td>
      <td>13.276030</td>
    </tr>
    <tr>
      <th>S</th>
      <td>449.527950</td>
      <td>0.336957</td>
      <td>2.350932</td>
      <td>29.445397</td>
      <td>0.571429</td>
      <td>0.413043</td>
      <td>27.079812</td>
    </tr>
  </tbody>
</table>
</div>



#### We observe a significantly better survival rate among passengers embarked in France (55%) than in UK (34% to 39%).
#### Although one may suggest that citizenship or language was a critical factor ( French are better swimmers ?), there are no  available data such as country-of-origin to investigate further.
#### A more careful look at the stats shows a discrepancy in the "Class" and the "Fare" between FR and UK embarkments:
- The Cherbourg-embarked were in higher class (mean 1.88) than Queenstown/Southampton (2.35 to 2.90)
- The Cherbourg-embarked paid a largely higher fare (59.95) than the UK-embarked (13.27 to 27.07)


```python
barplot('Embarked', 'Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb98f240>




![png](images/output_44_1.png)


#### The same comparison of survival by "Embarked" visualized in a barplot.


```python
sns.countplot(x='Survived', hue="Pclass", data=titanic_df, order=[0,1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc3288d0>




![png](images/output_46_1.png)


#### Survivors (1) vs Non-Survivors (0) according to "Pclass", in absolute values. The 3rd Class suffered the largest hit, almost 4 times the other classes.


```python
sns.barplot(x= 'Sex', y= 'Survived', hue="Pclass", data= titanic_df, ci= None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc610748>




![png](images/output_48_1.png)


#### A more detailled analysis of survival by Class and Sex, in percentage. Over 90% of female passengers in 1st and 2nd Class survived.


```python
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[0,1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc3285f8>




![png](images/output_50_1.png)


#### Survivors (1) vs Non-Survivors (0) according to "Embarked", in absolute values. Passengers embarked in Southampton suffered the largest hit, almost 9 times the other ports.


```python
sns.barplot(x= 'Sex', y= 'Survived', hue="Embarked", data= titanic_df, ci= None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xca66780>




![png](images/output_52_1.png)


#### A more detailled analysis of survival by Port and Sex, in percentage, gives a different outlook than the previous graph in absolute values.
#### While all ports for females had a similar impact on survival (70 to 85%), if you were a male to embark in Queenstown, you were doomed ! (less than 10% chance of survival).



## Data Wrangling Phase

#### As shown earlier, there are several columns missing data as NA (not available), the key one being "Age".
#### Also, some columns use a string format ("Sex" and "Embarked") which may complicate further calculations.
#### Let's see if we can fix these, starting with the easiest steps.

#### We now convert the "Sex" column from ('Female' or 'Male') into a new "Gender" column with numerical values ('0' or '1').


```python
titanic_df['Gender'] = 4
```


```python
titanic_df['Gender'] = titanic_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
```


```python
titanic_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Now we convert the "Embarked" column ('C' or 'Q' or 'S') into a new "Port' column with numerical values ('1' or '2' or '3').


```python
titanic_df['Port'] = 4
```


```python
titanic_df['Port'] = titanic_df['Embarked'].dropna().map( {'C': 1, 'Q': 2, 'S': 3} ).astype(int)
```


```python
titanic_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Gender</th>
      <th>Port</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Final step in data wrangling, we fix the missing "Age" values.
#### I suggest that we build an array to calculate the median age by "Sex" and "Pclass", then use those values to replace the missing ones in a new column named "Gender".
#### So we keep the original one "Age" intact, as opposed to forcing our way in with some *.pandas.df.fillna().


```python
median_ages = np.zeros((2,3))
median_ages
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = titanic_df[(titanic_df['Gender'] == i) & \
                              (titanic_df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages
```




    array([[ 35. ,  28. ,  21.5],
           [ 40. ,  30. ,  25. ]])



#### We created an array to calculate the median "Age" value by "Sex" and "Pclass".
#### Now we will use those to replace the missing ones in a new column "AgeFill", proceding step by step with due tests for checks.


```python
titanic_df['AgeFill'] = titanic_df['Age']
```


```python
# we check a sample before.
titanic_df[ titanic_df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We apply the replacing code.
for i in range(0, 2):
    for j in range(0, 3):
        titanic_df.loc[ (titanic_df.Age.isnull()) & (titanic_df.Gender == i) & (titanic_df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
```


```python
# We check the same sample after implementing the replacing code.
titanic_df[ titanic_df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>3</td>
      <td>NaN</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
</div>



#### This above is the survival rate by "Age", rounded to the interger value.


```python
fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(211)
titanic_df['AgeFill'].hist(bins=16, range=(0,80), alpha = .5).plot()
ax2 = fig.add_subplot(212)
titanic_df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5).plot()
```




    []




![png](images/output_72_1.png)


#### The revised histogram of ages (Up) from the column "AgeFill", with the original one (Low).
#### The new peak in 20:30 is artificially caused by the injection of 177 median values.

#### Now that we have an updated "Age" column with the "AgeFill" column, we can investigate the survival rate per age.
#### For better reading, we fix the comma/floating values by forcing an 'int' type.


```python
# convert from float to int
titanic_df['AgeFillInt'] = titanic_df['AgeFill'].astype(int)
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
barplot('AgeFillInt', 'Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xcfd7278>




![png](images/output_75_1.png)


#### This above is the survival rate by "Age", rounded to the interger value.


```python

```

### From now on, I calculate more granular analysis based on three variables, such as "Embarked" + "Sex" + "Pclass" to find the most potent combination for death or survival.
#### First we count the passengers for each combination then the relevant survival rate


```python
# Computing survival probability based on "Port", "Sex" and "Class".
max_chance= gb_mean(['Embarked', 'Sex', 'Pclass'])['Survived']
max_chance
```




    Embarked  Sex     Pclass
    C         female  1         0.976744
                      2         1.000000
                      3         0.652174
              male    1         0.404762
                      2         0.200000
                      3         0.232558
    Q         female  1         1.000000
                      2         1.000000
                      3         0.727273
              male    1         0.000000
                      2         0.000000
                      3         0.076923
    S         female  1         0.958333
                      2         0.910448
                      3         0.375000
              male    1         0.354430
                      2         0.154639
                      3         0.128302
    Name: Survived, dtype: float64




```python
titanic_df.groupby(['Embarked', 'Sex', 'Pclass'])['Survived'].count()
```




    Embarked  Sex     Pclass
    C         female  1          43
                      2           7
                      3          23
              male    1          42
                      2          10
                      3          43
    Q         female  1           1
                      2           2
                      3          33
              male    1           1
                      2           1
                      3          39
    S         female  1          48
                      2          67
                      3          88
              male    1          79
                      2          97
                      3         265
    Name: Survived, dtype: int64



#### Several observations:
- Some combinations had too few passengers (down to 1 only) to validate the survival rate
- Others seemed more realistic like "The 43 Females in 1st Class embarked in Cherbourg" with 97.7% survival or "The 265 Males in 3rd Class embarked in Southampton" with 12.8% survival

## Chi-Squared Test
#### We've seen in "Mutiple-variable exploration of -Survived-" that Sex seemed to have the highest impact in the survival rate of the passengers.
#### Let's run a Chi-Squared Test on the statistical significance of relationship between "Sex" and "Survived" variables.
- Probability of 0: both variable are dependant
- Probability of 1: both variables are independant
- Probability of less than 0.05: the relationship between the two variables is significant at 95% confidence


```python
def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(ctsum.fillna(0)))


chisq_of_df_cols(titanic_df, 'Sex', 'Survived')
```




    (260.71702016732104,
     1.1973570627755645e-58,
     1L,
     array([[ 193.47474747,  355.52525253],
            [ 120.52525253,  221.47474747]]))



#### Since the p-value < 0.05 (1.97 e-58), we can infer that Sex is a significant variable of Survived.

## CONCLUSION
### Limitations
- The dataset is incomplete as it includes only 891 passengers out of 2,224 in total, including 885 crews.
https://en.wikipedia.org/wiki/RMS_Titanic#Survivors_and_victims
- There were many missing values, including 177 "Ages" and over 680 "Cabin". We had to used median Age values par Sex and Pclass to fill in, thus skewing the data towards the medians.
- The Fares values included Grouped-Purchase without indication of individual prices, thus preventing a true analysis per fare.

### Summary
#### We found out that three variables had a strong impact on the survival rate of passengers:
- Sex: 74.2% of the Females survived while only 18.9% of the Males did.
- Pclass: 62.9% of the passengers in 1st Class survived vs. 47.3% in 2nd Class and 24.2% in 3rd Class.
- Embarked: 53.5% of the passengers embarked in Cherbourg survived vs. 38.9% in Queenstown and 33.7% in Southampton.

#### We also found two extreme cases of survival rate when combining the three variables:
- If you were a Female embarked in 1st Class in Cherbourg, your survival rate was highest with 97.67%
- If you were a Male embarked in 3rd Class in Queenstown, your survival rate was lowest with 7.69%

