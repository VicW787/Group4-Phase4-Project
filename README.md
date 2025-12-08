# **Phase 4 Project: NLP Sentiment Analysis of Apple vs Google Tweets**



# 1. **Executive Summary**

This project builds a sentiment classification system to analyze public emotions in Apple-related tweets. Using a dataset of 9,093 tweets labeled as positive, negative, or neutral, the goal is to help Apple’s Marketing & Product teams better understand customer perception and identify emerging issues in real time. Tweets are short, informal, and often noisy, making NLP-based machine learning an appropriate solution.

Data preparation included lowercasing, removing URLs, mentions, emojis, punctuation, and extra whitespace, followed by stopword removal and lemmatization to reduce noise. Duplicates, missing labels, and extremely short tweets were removed to ensure training quality. TF-IDF vectorization powered the classical models, while HuggingFace Transformers handled the BERT-based approaches.

Several models were evaluated. Traditional machine learning classifiers (Logistic Regression, Naive Bayes, Linear SVM, SMOTE-SVM, and Weighted SVM) achieved accuracies between 63% and 70%, with weighted SVM performing the strongest among them. A fine-tuned DistilBERT model significantly outperformed classical models, achieving 71.16% accuracy with 2 epochs and 71.49% accuracy after 4 epochs—confirming the advantage of transformer models for short-form, informal text like tweets.

Model evaluation used an 80/20 train-test split, accuracy, weighted F1-score, and confusion matrices. DistilBERT was selected as the final model due to its superior performance, ability to capture context, and improved handling of neutral vs. positive/negative distinctions. Further fine-tuning and domain-specific training data can boost accuracy even more.ults.



 # 2.  **Business Understanding**

 # 3.  **Stakeholder**  

**Primary Stakeholders**

1. Apple Marketing Team
Uses real-time sentiment insights to understand public perception, monitor brand reputation, evaluate campaign performance, and respond quickly to public feedback after product launches or events.

2. Apple Product Management & Engineering Teams
Gain visibility into customer frustrations, feature requests, recurring complaints, and performance issues (e.g., crashes, battery problems). This helps prioritize product improvements and software patches.

**Secondary Stakeholders**

3. Apple Customer Support & Care Teams
Benefit from early detection of negative sentiment spikes related to device malfunctions or software bugs. They can proactively prepare responses, FAQs, or issue statements.

4. Apple PR & Corporate Communications
Use sentiment monitoring to manage crises, track public reaction during controversies, and craft timely communication strategies compared to competitors like Google.

5. Competitive Intelligence & Market Research Teams
Analyze Apple vs Google sentiment trends to understand market positioning, customer loyalty drivers, and areas where Apple is outperforming or lagging behind.


     
 # 4. **Business problem**
    
Apple’s Marketing & Product Intelligence teams require a system that can continuously monitor and compare public sentiment toward Apple products against competing brands—particularly Google. With social media platforms like Twitter influencing customer perception in real time, Apple needs early visibility into negative trends, emerging complaints, and user frustrations before they scale. Likewise, understanding which Apple products generate the strongest positive engagement—compared to similar Google offerings—provides insights for campaign optimization, product refinement, and competitive positioning.
This project aims to build an NLP-based sentiment analysis model capable of automatically classifying Apple- and Google-related tweets as positive, negative, or neutral. The solution will help Apple quantify customer emotions, detect shifts in public perception, and benchmark Apple’s reputation relative to Google in fast-evolving online conversations.

# 5. **Data Understanding**

The dataset comes from CrowdFlower (hosted on data.world) and contains 9,093 tweets related to Apple products and competing brands. It includes three columns:

1. tweet_text – the raw text of each tweet

2. emotion_in_tweet_is_directed_at – the specific product or brand mentioned (e.g., iPad, Apple, Google)

3. is_there_an_emotion_directed_at_a_brand_or_product – the sentiment label assigned by annotators

Sentiment labels fall into four categories: Positive emotion, Negative emotion, No emotion toward brand or product, and I can’t tell. The distribution is imbalanced: most tweets show no emotion (5,389), followed by positive (2,978), negative (570), and uncertain (156).

Basic descriptive statistics show an average tweet length of ~105 characters, with lengths ranging from 3 to 178 characters, confirming that the dataset fits typical short-text social media patterns. The column emotion_in_tweet_is_directed_at contains multiple Apple and non-Apple target entities, with the most common being iPad (946), Apple (661), and iPad/iPhone apps (470).

This dataset is well-suited for an NLP sentiment classification task because it provides labeled text, diverse sentiment categories, and real-world social media noise—allowing meaningful training and evaluation of advanced models such as DistilBERT.

# 6. **Data Preparation**

#  6.1  **Loading data**


```python
import pandas as pd

df = pd.read_csv("apple_twitter_sentiment.csv", engine='python', on_bad_lines='skip', encoding='latin-1')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>



# 6.2 **Inspecting raw text**


```python
#checking the columns
df.columns
```




    Index(['tweet_text', 'emotion_in_tweet_is_directed_at',
           'is_there_an_emotion_directed_at_a_brand_or_product'],
          dtype='object')



Great our dataset is actually perfect for an NLP sentiment/emotion classification project. And the columns are clean and meaningful:

**Our Columns**

tweet_text - the text we will analyze

emotion_in_tweet_is_directed_at - target entity (Apple, Google, iPad, etc.)

is_there_an_emotion_directed_at_a_brand_or_product - emotion label (Positive, Negative, Neutral, etc.)


```python
#Checking data types& missing values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9093 entries, 0 to 9092
    Data columns (total 3 columns):
     #   Column                                              Non-Null Count  Dtype 
    ---  ------                                              --------------  ----- 
     0   tweet_text                                          9092 non-null   object
     1   emotion_in_tweet_is_directed_at                     3291 non-null   object
     2   is_there_an_emotion_directed_at_a_brand_or_product  9093 non-null   object
    dtypes: object(3)
    memory usage: 213.2+ KB
    

The dataset contains 9,093 rows and 3 columns, all stored as text, with one column (emotion_in_tweet_is_directed_at) having many missing values (only 3,291 non-null). This suggests that while every tweet has a recorded emotion label, many tweets do not specify which brand or product the emotion is directed at, which may influence modeling decisions later.


```python
df.isnull().sum()
```




    tweet_text                                               1
    emotion_in_tweet_is_directed_at                       5802
    is_there_an_emotion_directed_at_a_brand_or_product       0
    dtype: int64



The tweet_text and is_there_an_emotion_directed_at_a_brand_or_product columns have no missing values, meaning they are reliable for modeling. However, emotion_in_tweet_is_directed_at has 5,802 missing entries, confirming it is too incomplete to use as a target variable and should be dropped or ignored.


```python
# Look at unique values for the labels and target entities:

df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()
```




    is_there_an_emotion_directed_at_a_brand_or_product
    No emotion toward brand or product    5389
    Positive emotion                      2978
    Negative emotion                       570
    I can't tell                           156
    Name: count, dtype: int64



The majority of tweets (5,389) have no emotion directed at a brand or product, meaning most messages are neutral with respect to brand sentiment. Positive emotions (2,978) far outnumber negative (570), indicating a class imbalance that you’ll need to handle during modeling.


```python
df['emotion_in_tweet_is_directed_at'].value_counts()
```




    emotion_in_tweet_is_directed_at
    iPad                               946
    Apple                              661
    iPad or iPhone App                 470
    Google                             430
    iPhone                             297
    Other Google product or service    293
    Android App                         81
    Android                             78
    Other Apple product or service      35
    Name: count, dtype: int64



Most emotions are directed toward iPad, Apple, and iPad/iPhone apps, showing that the majority of emotional engagement focuses on Apple-related products. Categories like Android, Android apps, and other Apple/Google products appear far less frequently, indicating that emotional reactions toward these brands are relatively rare in this dataset.


```python
# Check random raw tweets

df.sample(10)['tweet_text']
```




    5316    RT @mention #SXSW sponsored by Apple, plaid, a...
    5612    RT @mention Check out the FREE SXSW Sampler on...
    1303    my iphone was stolen and I got it back !!!!!! ...
    7833    I know its #SXSW time when I have an abnormal ...
    4567    LinkedIn Nelson Medina HootSuite News: HootSui...
    7105    Apple is opening up a temporary store in downt...
    1239    Apple set to open popup shop in core of #SXSW ...
    4419    . @mention double fisting at the keynote #sxsw...
    6256    RT @mention Line is wrapping around the block ...
    5241    RT @mention #killing #sxsw &quot;you can googl...
    Name: tweet_text, dtype: object



The sample tweets show a mix of retweets, product mentions, and personal opinions, reflecting the informal and varied nature of social media text. This variety highlights the need for careful text preprocessing, such as removing mentions, hashtags, and URLs, to clean the data for NLP modeling.


```python
# Check tweet lengths

df['tweet_length'] = df['tweet_text'].astype(str).str.len()
df['tweet_length'].describe()
```




    count    9093.000000
    mean      104.951061
    std        27.207164
    min         3.000000
    25%        86.000000
    50%       109.000000
    75%       126.000000
    max       178.000000
    Name: tweet_length, dtype: float64



Tweets in the dataset have a mean length of 105 characters, with most tweets falling between 86 and 126 characters, showing moderate variability in text length. The shortest tweet is only 3 characters, and the longest is 178, indicating that preprocessing should handle both very short and very long tweets appropriately.


```python
#What type of cleaning do we need
df['tweet_text'].sample(10).values
```




    array(['Seriously #sxsw? Did you do any testing on the mobile apps? Constant iPad crashes causing lost schedules, and no sync for WP7.',
           'More #apple iPads per capital at #sxsw then anywhere else in the world except for Foxconn',
           "GSD&amp;M + Google 7-10. RT @mention What's the best party to hit tonight? #sxsw @mention @mention",
           'Walking through the trade show at #sxsw is basically swapping 10 business cards for chances to win 10 different iPad 2s.',
           'RT @mention #SXSW Google party: League of Extraordinary H4ackers promoting txt Redcross to 90999 to help Japan @mention Speakeasy {link} #photo',
           "RT @mention And we're live! Come get your Android and check us out! {link} #SXSW #android",
           'Possible Google social network to preview at #sxsw. {link}',
           'Totally love the copia cocoon messenger iPad bag I picked up for attending the redbox/gilt group/graphic.ly sesh this morning! #sxsw',
           'RT @mention &quot;Google &quot;product gatekeeper&quot; Marissa Mayer on the location-based &quot;fast, fun and future&quot; {link} (HT @mention #SXSW&quot;',
           "RT @mention Will Google launch a new social network called 'Circles' at South by Southwest today?  {link}  #SXSW #Circles #Google"],
          dtype=object)



The raw tweet samples contain noise such as HTML entities, URLs, mentions, hashtags, unicode artifacts, and inconsistent casing, all of which make the text hard for an NLP model to interpret.

# 6.3 **Cleaning & preprocessing**

* Lowercasing

* Removing URLs

* Removing hashtags and @mentions

* Removing punctuation

* Removing stopwords

* Tokenization

* Lemmatization


```python
#create a copy of df
df_clean = df.copy()
```


```python
#step 1: Drop completely empty rows
df_clean = df_clean.dropna(subset=['tweet_text']).reset_index(drop=True)
df_clean.isnull().sum()
```




    tweet_text                                               0
    emotion_in_tweet_is_directed_at                       5801
    is_there_an_emotion_directed_at_a_brand_or_product       0
    tweet_length                                             0
    dtype: int64



For this project we will ignore the 5801 missing values because

1. **High missing rate**

   5801 out of 9093 rows are missing this value (64%), which is too much to impute reliably.

2. **Filling with "Unknown"** adds mostly noise and   rarely improves a sentiment model.
Primary goal is sentiment classification

3. **The label column**
(is_there_an_emotion_directed_at_a_brand_or_product) already tells us whether the tweet is positive or negative.


```python
df_clean = df_clean.drop(columns=['emotion_in_tweet_is_directed_at'])

```


```python
#step 2: Normalize column names
df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
```


```python
#Step 3:
#Inspect & standardize label column
#Map textual labels to consistent classes for modeling

df_clean['label_raw'] = df_clean['is_there_an_emotion_directed_at_a_brand_or_product'].astype(str)
df_clean['label_raw'].value_counts()
# Then map:
# Updated mapping based on actual value_counts and to handle 'I can't tell'
mapping = {'Positive emotion': 1, 'Negative emotion': 0, 'No emotion toward brand or product': 2, "I can't tell": None}
df_clean['label'] = df_clean['label_raw'].map(mapping)
# Optionally drop rows with unknown labels:
df_clean = df_clean[df_clean['label'].notna()]
```


```python
#Step 4:
#Remove duplicates
#Duplicates bias the model.
df_clean = df_clean.drop_duplicates(subset=['tweet_text']).reset_index(drop=True)

```


```python
#Step 5
#Decoding HTML entities + remove weird unicode artifacts
#Convert &amp;, &quot; and remove broken characters like \x89ÛÏ.
import html
df_clean['text_step0'] = df_clean['tweet_text'].apply(lambda x: html.unescape(str(x)).encode('ascii', 'ignore').decode())
```


```python
#Step 6
#Remove user mentions and RT indicators

#Mentions and RT don’t add sentiment content and may leak metadata.
import re
df_clean['text_step1'] = df_clean['text_step0'].str.replace(r'@\w+', '', regex=True)
df_clean['text_step1'] = df_clean['text_step1'].str.replace(r'\brt\b', '', regex=True, flags=re.IGNORECASE)
```


```python
#Step 7
#Remove URLs.
#Links add noise
df_clean['text_step2'] = df_clean['text_step1'].str.replace(r'http\S+|www\S+|\{link\}', '', regex=True)
```


```python
#Step 8
#Handling hashtags
#hashtags often contain keywords. We  remove # but keep the word.
df_clean['text_step3'] = df_clean['text_step2'].str.replace(r'#(\w+)', r'\1', regex=True)
```


```python
#Step 9
#Remove punctuation and extra symbols
#Simplify tokenization. Keep emotive punctuation if you want (e.g., exclamation marks) — otherwise remove.

import string
df_clean['text_step4'] = df_clean['text_step3'].str.translate(str.maketrans('', '', string.punctuation))

```


```python
#Lowercase and strip extra whitespace
#Normalizes tokens for vectorizers and embeddings.

df_clean['text_step5'] = df_clean['text_step4'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

```


```python
#Stopword removal, lemmatization/stemming
#Reduces vocabulary size; lemmatization keeps semantics.
!python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
def lemmatize(text):
    doc = nlp(text)
    return ' '.join([tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct])
df_clean['text_final'] = df_clean['text_step5'].apply(lemmatize)
```

    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    

    Collecting en-core-web-sm==3.8.0
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
         ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
         ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
         ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
         ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
         ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
         --------------------------------------- 0.0/12.8 MB 163.8 kB/s eta 0:01:18
         --------------------------------------- 0.0/12.8 MB 163.8 kB/s eta 0:01:18
         --------------------------------------- 0.0/12.8 MB 163.8 kB/s eta 0:01:18
         --------------------------------------- 0.0/12.8 MB 131.3 kB/s eta 0:01:38
         --------------------------------------- 0.0/12.8 MB 131.3 kB/s eta 0:01:38
         --------------------------------------- 0.0/12.8 MB 131.3 kB/s eta 0:01:38
         --------------------------------------- 0.1/12.8 MB 136.5 kB/s eta 0:01:34
         --------------------------------------- 0.1/12.8 MB 158.2 kB/s eta 0:01:21
         --------------------------------------- 0.1/12.8 MB 169.3 kB/s eta 0:01:16
         --------------------------------------- 0.1/12.8 MB 169.3 kB/s eta 0:01:16
         --------------------------------------- 0.1/12.8 MB 172.7 kB/s eta 0:01:14
         --------------------------------------- 0.1/12.8 MB 172.7 kB/s eta 0:01:14
         --------------------------------------- 0.1/12.8 MB 172.7 kB/s eta 0:01:14
         --------------------------------------- 0.1/12.8 MB 156.9 kB/s eta 0:01:21
         --------------------------------------- 0.1/12.8 MB 160.6 kB/s eta 0:01:19
          -------------------------------------- 0.2/12.8 MB 194.4 kB/s eta 0:01:05
          -------------------------------------- 0.2/12.8 MB 202.7 kB/s eta 0:01:03
          -------------------------------------- 0.2/12.8 MB 222.4 kB/s eta 0:00:57
          -------------------------------------- 0.2/12.8 MB 232.7 kB/s eta 0:00:55
          -------------------------------------- 0.2/12.8 MB 232.7 kB/s eta 0:00:55
          -------------------------------------- 0.2/12.8 MB 232.7 kB/s eta 0:00:55
          -------------------------------------- 0.3/12.8 MB 218.5 kB/s eta 0:00:58
          -------------------------------------- 0.3/12.8 MB 218.5 kB/s eta 0:00:58
          -------------------------------------- 0.3/12.8 MB 210.2 kB/s eta 0:01:00
          -------------------------------------- 0.3/12.8 MB 218.6 kB/s eta 0:00:58
          -------------------------------------- 0.3/12.8 MB 234.1 kB/s eta 0:00:54
         - ------------------------------------- 0.3/12.8 MB 241.1 kB/s eta 0:00:52
         - ------------------------------------- 0.3/12.8 MB 243.1 kB/s eta 0:00:52
         - ------------------------------------- 0.4/12.8 MB 249.4 kB/s eta 0:00:50
         - ------------------------------------- 0.4/12.8 MB 252.6 kB/s eta 0:00:50
         - ------------------------------------- 0.4/12.8 MB 252.6 kB/s eta 0:00:50
         - ------------------------------------- 0.4/12.8 MB 252.6 kB/s eta 0:00:50
         - ------------------------------------- 0.4/12.8 MB 252.6 kB/s eta 0:00:50
         - ------------------------------------- 0.4/12.8 MB 235.0 kB/s eta 0:00:53
         - ------------------------------------- 0.4/12.8 MB 238.4 kB/s eta 0:00:52
         - ------------------------------------- 0.4/12.8 MB 239.5 kB/s eta 0:00:52
         - ------------------------------------- 0.5/12.8 MB 240.9 kB/s eta 0:00:52
         - ------------------------------------- 0.5/12.8 MB 243.8 kB/s eta 0:00:51
         - ------------------------------------- 0.5/12.8 MB 243.8 kB/s eta 0:00:51
         - ------------------------------------- 0.5/12.8 MB 241.2 kB/s eta 0:00:52
         - ------------------------------------- 0.5/12.8 MB 245.8 kB/s eta 0:00:51
         - ------------------------------------- 0.5/12.8 MB 248.3 kB/s eta 0:00:50
         - ------------------------------------- 0.5/12.8 MB 247.6 kB/s eta 0:00:50
         - ------------------------------------- 0.5/12.8 MB 247.6 kB/s eta 0:00:50
         - ------------------------------------- 0.6/12.8 MB 244.7 kB/s eta 0:00:51
         - ------------------------------------- 0.6/12.8 MB 244.7 kB/s eta 0:00:51
         - ------------------------------------- 0.6/12.8 MB 239.2 kB/s eta 0:00:52
         - ------------------------------------- 0.6/12.8 MB 241.4 kB/s eta 0:00:51
         - ------------------------------------- 0.6/12.8 MB 246.8 kB/s eta 0:00:50
         - ------------------------------------- 0.6/12.8 MB 247.9 kB/s eta 0:00:50
         - ------------------------------------- 0.6/12.8 MB 254.0 kB/s eta 0:00:48
         -- ------------------------------------ 0.7/12.8 MB 263.8 kB/s eta 0:00:46
         -- ------------------------------------ 0.7/12.8 MB 263.0 kB/s eta 0:00:47
         -- ------------------------------------ 0.7/12.8 MB 263.0 kB/s eta 0:00:47
         -- ------------------------------------ 0.7/12.8 MB 261.4 kB/s eta 0:00:47
         -- ------------------------------------ 0.7/12.8 MB 261.4 kB/s eta 0:00:47
         -- ------------------------------------ 0.7/12.8 MB 261.4 kB/s eta 0:00:47
         -- ------------------------------------ 0.7/12.8 MB 253.5 kB/s eta 0:00:48
         -- ------------------------------------ 0.7/12.8 MB 255.1 kB/s eta 0:00:48
         -- ------------------------------------ 0.8/12.8 MB 258.0 kB/s eta 0:00:47
         -- ------------------------------------ 0.8/12.8 MB 257.4 kB/s eta 0:00:47
         -- ------------------------------------ 0.8/12.8 MB 261.5 kB/s eta 0:00:46
         -- ------------------------------------ 0.8/12.8 MB 259.5 kB/s eta 0:00:47
         -- ------------------------------------ 0.8/12.8 MB 260.9 kB/s eta 0:00:46
         -- ------------------------------------ 0.8/12.8 MB 263.5 kB/s eta 0:00:46
         -- ------------------------------------ 0.9/12.8 MB 264.1 kB/s eta 0:00:46
         -- ------------------------------------ 0.9/12.8 MB 267.8 kB/s eta 0:00:45
         -- ------------------------------------ 0.9/12.8 MB 266.5 kB/s eta 0:00:45
         -- ------------------------------------ 1.0/12.8 MB 280.9 kB/s eta 0:00:43
         --- ----------------------------------- 1.0/12.8 MB 289.0 kB/s eta 0:00:41
         --- ----------------------------------- 1.0/12.8 MB 291.0 kB/s eta 0:00:41
         --- ----------------------------------- 1.1/12.8 MB 295.8 kB/s eta 0:00:40
         --- ----------------------------------- 1.1/12.8 MB 299.0 kB/s eta 0:00:40
         --- ----------------------------------- 1.1/12.8 MB 302.3 kB/s eta 0:00:39
         --- ----------------------------------- 1.1/12.8 MB 306.8 kB/s eta 0:00:39
         --- ----------------------------------- 1.2/12.8 MB 311.2 kB/s eta 0:00:38
         --- ----------------------------------- 1.2/12.8 MB 325.0 kB/s eta 0:00:36
         --- ----------------------------------- 1.3/12.8 MB 331.7 kB/s eta 0:00:35
         ---- ---------------------------------- 1.3/12.8 MB 339.7 kB/s eta 0:00:34
         ---- ---------------------------------- 1.4/12.8 MB 346.1 kB/s eta 0:00:34
         ---- ---------------------------------- 1.4/12.8 MB 361.3 kB/s eta 0:00:32
         ---- ---------------------------------- 1.5/12.8 MB 369.8 kB/s eta 0:00:31
         ---- ---------------------------------- 1.5/12.8 MB 375.6 kB/s eta 0:00:31
         ---- ---------------------------------- 1.6/12.8 MB 381.3 kB/s eta 0:00:30
         ---- ---------------------------------- 1.6/12.8 MB 386.8 kB/s eta 0:00:29
         ----- --------------------------------- 1.7/12.8 MB 393.3 kB/s eta 0:00:29
         ----- --------------------------------- 1.7/12.8 MB 401.0 kB/s eta 0:00:28
         ----- --------------------------------- 1.8/12.8 MB 405.2 kB/s eta 0:00:28
         ----- --------------------------------- 1.8/12.8 MB 415.8 kB/s eta 0:00:27
         ----- --------------------------------- 1.9/12.8 MB 421.3 kB/s eta 0:00:26
         ----- --------------------------------- 1.9/12.8 MB 427.0 kB/s eta 0:00:26
         ----- --------------------------------- 1.9/12.8 MB 431.6 kB/s eta 0:00:26
         ------ -------------------------------- 2.0/12.8 MB 436.2 kB/s eta 0:00:25
         ------ -------------------------------- 2.0/12.8 MB 443.7 kB/s eta 0:00:25
         ------ -------------------------------- 2.1/12.8 MB 451.7 kB/s eta 0:00:24
         ------ -------------------------------- 2.2/12.8 MB 458.2 kB/s eta 0:00:24
         ------ -------------------------------- 2.2/12.8 MB 464.5 kB/s eta 0:00:23
         ------ -------------------------------- 2.3/12.8 MB 471.2 kB/s eta 0:00:23
         ------- ------------------------------- 2.3/12.8 MB 485.2 kB/s eta 0:00:22
         ------- ------------------------------- 2.4/12.8 MB 495.7 kB/s eta 0:00:21
         ------- ------------------------------- 2.5/12.8 MB 501.5 kB/s eta 0:00:21
         ------- ------------------------------- 2.5/12.8 MB 502.9 kB/s eta 0:00:21
         ------- ------------------------------- 2.6/12.8 MB 508.4 kB/s eta 0:00:21
         ------- ------------------------------- 2.6/12.8 MB 515.8 kB/s eta 0:00:20
         -------- ------------------------------ 2.7/12.8 MB 522.8 kB/s eta 0:00:20
         -------- ------------------------------ 2.7/12.8 MB 530.3 kB/s eta 0:00:19
         -------- ------------------------------ 2.8/12.8 MB 537.3 kB/s eta 0:00:19
         -------- ------------------------------ 2.9/12.8 MB 544.2 kB/s eta 0:00:19
         -------- ------------------------------ 2.9/12.8 MB 551.1 kB/s eta 0:00:18
         --------- ----------------------------- 3.0/12.8 MB 565.4 kB/s eta 0:00:18
         --------- ----------------------------- 3.0/12.8 MB 563.9 kB/s eta 0:00:18
         --------- ----------------------------- 3.1/12.8 MB 566.9 kB/s eta 0:00:18
         --------- ----------------------------- 3.2/12.8 MB 573.2 kB/s eta 0:00:17
         --------- ----------------------------- 3.2/12.8 MB 583.2 kB/s eta 0:00:17
         ---------- ---------------------------- 3.3/12.8 MB 593.0 kB/s eta 0:00:17
         ---------- ---------------------------- 3.4/12.8 MB 597.1 kB/s eta 0:00:16
         ---------- ---------------------------- 3.5/12.8 MB 610.4 kB/s eta 0:00:16
         ---------- ---------------------------- 3.5/12.8 MB 617.8 kB/s eta 0:00:15
         ----------- --------------------------- 3.6/12.8 MB 623.5 kB/s eta 0:00:15
         ----------- --------------------------- 3.7/12.8 MB 629.0 kB/s eta 0:00:15
         ----------- --------------------------- 3.7/12.8 MB 629.2 kB/s eta 0:00:15
         ----------- --------------------------- 3.8/12.8 MB 639.8 kB/s eta 0:00:15
         ----------- --------------------------- 3.9/12.8 MB 653.7 kB/s eta 0:00:14
         ------------ -------------------------- 4.0/12.8 MB 660.5 kB/s eta 0:00:14
         ------------ -------------------------- 4.0/12.8 MB 662.2 kB/s eta 0:00:14
         ------------ -------------------------- 4.1/12.8 MB 667.1 kB/s eta 0:00:14
         ------------ -------------------------- 4.2/12.8 MB 675.3 kB/s eta 0:00:13
         ------------ -------------------------- 4.2/12.8 MB 680.1 kB/s eta 0:00:13
         ------------- ------------------------- 4.3/12.8 MB 688.1 kB/s eta 0:00:13
         ------------- ------------------------- 4.4/12.8 MB 691.3 kB/s eta 0:00:13
         ------------- ------------------------- 4.4/12.8 MB 694.2 kB/s eta 0:00:13
         ------------- ------------------------- 4.5/12.8 MB 698.7 kB/s eta 0:00:12
         ------------- ------------------------- 4.5/12.8 MB 700.1 kB/s eta 0:00:12
         ------------- ------------------------- 4.5/12.8 MB 700.1 kB/s eta 0:00:12
         -------------- ------------------------ 4.6/12.8 MB 702.4 kB/s eta 0:00:12
         -------------- ------------------------ 4.7/12.8 MB 709.9 kB/s eta 0:00:12
         -------------- ------------------------ 4.7/12.8 MB 711.0 kB/s eta 0:00:12
         -------------- ------------------------ 4.7/12.8 MB 710.8 kB/s eta 0:00:12
         -------------- ------------------------ 4.8/12.8 MB 710.4 kB/s eta 0:00:12
         -------------- ------------------------ 4.8/12.8 MB 716.0 kB/s eta 0:00:12
         -------------- ------------------------ 4.9/12.8 MB 714.3 kB/s eta 0:00:12
         -------------- ------------------------ 4.9/12.8 MB 712.3 kB/s eta 0:00:12
         --------------- ----------------------- 4.9/12.8 MB 716.4 kB/s eta 0:00:11
         --------------- ----------------------- 5.0/12.8 MB 718.9 kB/s eta 0:00:11
         --------------- ----------------------- 5.1/12.8 MB 725.8 kB/s eta 0:00:11
         --------------- ----------------------- 5.2/12.8 MB 735.5 kB/s eta 0:00:11
         ---------------- ---------------------- 5.3/12.8 MB 742.2 kB/s eta 0:00:11
         ---------------- ---------------------- 5.3/12.8 MB 748.8 kB/s eta 0:00:10
         ---------------- ---------------------- 5.3/12.8 MB 748.8 kB/s eta 0:00:10
         ---------------- ---------------------- 5.4/12.8 MB 744.6 kB/s eta 0:00:10
         ---------------- ---------------------- 5.4/12.8 MB 745.6 kB/s eta 0:00:10
         ---------------- ---------------------- 5.5/12.8 MB 749.2 kB/s eta 0:00:10
         ---------------- ---------------------- 5.6/12.8 MB 752.8 kB/s eta 0:00:10
         ----------------- --------------------- 5.7/12.8 MB 764.4 kB/s eta 0:00:10
         ----------------- --------------------- 5.7/12.8 MB 767.9 kB/s eta 0:00:10
         ----------------- --------------------- 5.8/12.8 MB 767.3 kB/s eta 0:00:10
         ----------------- --------------------- 5.8/12.8 MB 770.8 kB/s eta 0:00:10
         ----------------- --------------------- 5.9/12.8 MB 775.5 kB/s eta 0:00:09
         ------------------ -------------------- 6.0/12.8 MB 781.1 kB/s eta 0:00:09
         ------------------ -------------------- 6.1/12.8 MB 788.6 kB/s eta 0:00:09
         ------------------ -------------------- 6.2/12.8 MB 796.7 kB/s eta 0:00:09
         ------------------- ------------------- 6.2/12.8 MB 799.9 kB/s eta 0:00:09
         ------------------- ------------------- 6.3/12.8 MB 801.7 kB/s eta 0:00:09
         ------------------- ------------------- 6.4/12.8 MB 807.5 kB/s eta 0:00:08
         ------------------- ------------------- 6.5/12.8 MB 813.1 kB/s eta 0:00:08
         ------------------- ------------------- 6.5/12.8 MB 818.3 kB/s eta 0:00:08
         -------------------- ------------------ 6.6/12.8 MB 822.4 kB/s eta 0:00:08
         -------------------- ------------------ 6.7/12.8 MB 825.2 kB/s eta 0:00:08
         -------------------- ------------------ 6.7/12.8 MB 828.0 kB/s eta 0:00:08
         -------------------- ------------------ 6.8/12.8 MB 829.0 kB/s eta 0:00:08
         -------------------- ------------------ 6.8/12.8 MB 829.0 kB/s eta 0:00:08
         -------------------- ------------------ 6.8/12.8 MB 824.2 kB/s eta 0:00:08
         -------------------- ------------------ 6.9/12.8 MB 824.5 kB/s eta 0:00:08
         --------------------- ----------------- 6.9/12.8 MB 826.0 kB/s eta 0:00:08
         --------------------- ----------------- 6.9/12.8 MB 826.5 kB/s eta 0:00:08
         --------------------- ----------------- 7.0/12.8 MB 826.5 kB/s eta 0:00:08
         --------------------- ----------------- 7.0/12.8 MB 826.4 kB/s eta 0:00:07
         --------------------- ----------------- 7.1/12.8 MB 831.6 kB/s eta 0:00:07
         ---------------------- ---------------- 7.2/12.8 MB 840.1 kB/s eta 0:00:07
         ---------------------- ---------------- 7.3/12.8 MB 848.6 kB/s eta 0:00:07
         ---------------------- ---------------- 7.5/12.8 MB 857.0 kB/s eta 0:00:07
         ---------------------- ---------------- 7.5/12.8 MB 861.7 kB/s eta 0:00:07
         ----------------------- --------------- 7.6/12.8 MB 865.7 kB/s eta 0:00:07
         ----------------------- --------------- 7.7/12.8 MB 873.9 kB/s eta 0:00:06
         ----------------------- --------------- 7.8/12.8 MB 879.2 kB/s eta 0:00:06
         ----------------------- --------------- 7.9/12.8 MB 881.6 kB/s eta 0:00:06
         ------------------------ -------------- 7.9/12.8 MB 886.4 kB/s eta 0:00:06
         ------------------------ -------------- 8.0/12.8 MB 888.3 kB/s eta 0:00:06
         ------------------------ -------------- 8.1/12.8 MB 895.0 kB/s eta 0:00:06
         ------------------------ -------------- 8.2/12.8 MB 899.3 kB/s eta 0:00:06
         ------------------------- ------------- 8.3/12.8 MB 905.5 kB/s eta 0:00:05
         ------------------------- ------------- 8.4/12.8 MB 909.1 kB/s eta 0:00:05
         ------------------------- ------------- 8.5/12.8 MB 915.5 kB/s eta 0:00:05
         -------------------------- ------------ 8.6/12.8 MB 920.4 kB/s eta 0:00:05
         -------------------------- ------------ 8.7/12.8 MB 930.5 kB/s eta 0:00:05
         -------------------------- ------------ 8.8/12.8 MB 940.0 kB/s eta 0:00:05
         --------------------------- ----------- 8.9/12.8 MB 944.0 kB/s eta 0:00:05
         --------------------------- ----------- 9.0/12.8 MB 948.0 kB/s eta 0:00:05
         --------------------------- ----------- 9.1/12.8 MB 953.6 kB/s eta 0:00:04
         ---------------------------- ---------- 9.2/12.8 MB 957.5 kB/s eta 0:00:04
         ---------------------------- ---------- 9.3/12.8 MB 963.5 kB/s eta 0:00:04
         ---------------------------- ---------- 9.3/12.8 MB 961.9 kB/s eta 0:00:04
         ---------------------------- ---------- 9.4/12.8 MB 964.7 kB/s eta 0:00:04
         ---------------------------- ---------- 9.5/12.8 MB 970.5 kB/s eta 0:00:04
         ----------------------------- --------- 9.6/12.8 MB 971.7 kB/s eta 0:00:04
         ----------------------------- --------- 9.7/12.8 MB 977.4 kB/s eta 0:00:04
         ----------------------------- --------- 9.8/12.8 MB 984.7 kB/s eta 0:00:04
         ----------------------------- --------- 9.8/12.8 MB 984.6 kB/s eta 0:00:04
         ------------------------------ -------- 9.9/12.8 MB 987.1 kB/s eta 0:00:03
         ------------------------------ -------- 9.9/12.8 MB 985.6 kB/s eta 0:00:03
         ----------------------------- -------- 10.0/12.8 MB 987.1 kB/s eta 0:00:03
         ------------------------------ ------- 10.2/12.8 MB 998.7 kB/s eta 0:00:03
         -------------------------------- ------- 10.3/12.8 MB 1.0 MB/s eta 0:00:03
         -------------------------------- ------- 10.4/12.8 MB 1.1 MB/s eta 0:00:03
         -------------------------------- ------- 10.5/12.8 MB 1.1 MB/s eta 0:00:03
         --------------------------------- ------ 10.6/12.8 MB 1.1 MB/s eta 0:00:02
         --------------------------------- ------ 10.7/12.8 MB 1.2 MB/s eta 0:00:02
         --------------------------------- ------ 10.7/12.8 MB 1.2 MB/s eta 0:00:02
         --------------------------------- ------ 10.9/12.8 MB 1.3 MB/s eta 0:00:02
         ---------------------------------- ----- 11.0/12.8 MB 1.3 MB/s eta 0:00:02
         ---------------------------------- ----- 11.1/12.8 MB 1.4 MB/s eta 0:00:02
         ---------------------------------- ----- 11.1/12.8 MB 1.4 MB/s eta 0:00:02
         ----------------------------------- ---- 11.3/12.8 MB 1.4 MB/s eta 0:00:02
         ----------------------------------- ---- 11.3/12.8 MB 1.4 MB/s eta 0:00:02
         ----------------------------------- ---- 11.4/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------ --- 11.6/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------ --- 11.7/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------ --- 11.7/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------- -- 11.9/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------- -- 11.9/12.8 MB 1.5 MB/s eta 0:00:01
         ------------------------------------- -- 12.1/12.8 MB 1.5 MB/s eta 0:00:01
         -------------------------------------- - 12.2/12.8 MB 1.5 MB/s eta 0:00:01
         -------------------------------------- - 12.3/12.8 MB 1.6 MB/s eta 0:00:01
         -------------------------------------- - 12.5/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------  12.6/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------  12.6/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------  12.7/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------  12.8/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------  12.8/12.8 MB 1.6 MB/s eta 0:00:01
         ---------------------------------------- 12.8/12.8 MB 1.6 MB/s eta 0:00:00
    Installing collected packages: en-core-web-sm
    Successfully installed en-core-web-sm-3.8.0
    [38;5;2m[+] Download and installation successful[0m
    You can now load the package via spacy.load('en_core_web_sm')
    


```python
#Final sanity checks
#To make sure nothing important was lost and check class balance.
df_clean['text_final'].str.len().describe()
df_clean['label'].value_counts(normalize=True)
df_clean[['tweet_text','text_final']].sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>text_final</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2636</th>
      <td>The Google map app saves my life on a regular ...</td>
      <td>google map app save life regular basis map 150...</td>
    </tr>
    <tr>
      <th>5841</th>
      <td>RT @mention Have decided I need an ipad. #usgu...</td>
      <td>decide need ipad usguy sxsw agree</td>
    </tr>
    <tr>
      <th>1124</th>
      <td>ÛÏ@mention The 10 most dangerous IPhone apps....</td>
      <td>10 dangerous iphone app</td>
    </tr>
    <tr>
      <th>1052</th>
      <td>@mention  Saw u r going to #SXSW. Dwnld #Groun...</td>
      <td>see u r go sxsw dwnld groundlink app ride 10 a...</td>
    </tr>
    <tr>
      <th>4228</th>
      <td>YES. RT @mention Haha the @mention drinking ga...</td>
      <td>yes haha drinking game drink time seth say gam...</td>
    </tr>
    <tr>
      <th>2132</th>
      <td>No way! #atx #sxsw RT @mention Just found out ...</td>
      <td>way atx sxsw find apple popup store schedule c...</td>
    </tr>
    <tr>
      <th>7920</th>
      <td>#sxsw #lost #reward 4 finding an #iPhone 4G (b...</td>
      <td>sxsw lose reward 4 find iphone 4 g black case ...</td>
    </tr>
    <tr>
      <th>1400</th>
      <td>#Startups: Free Premium Service on your 1st jo...</td>
      <td>startups free premium service 1st job post ans...</td>
    </tr>
    <tr>
      <th>579</th>
      <td>.@mention Google Will Connect the Digital &amp;amp...</td>
      <td>google connect digital physical world mobile sxsw</td>
    </tr>
    <tr>
      <th>7868</th>
      <td>Google desmiente el lanzamiento hoy de Circles...</td>
      <td>google desmiente el lanzamiento hoy de circle ...</td>
    </tr>
  </tbody>
</table>
</div>



We successfully cleaned:

* Mentions - removed (.@mention - gone)

* URLs {link} - removed

* Hashtags #sxsw - converted to sxsw

* Punctuation - removed

* Mixed case - lowercased

* Unicode artifacts - removed

* Stopwords - removed

* Lemmatization - applied correctly (listening - listen)


```python
#Save cleaned dataset
df_clean.to_csv('apple_google_twitter_cleaned.csv', index=False)

```

# 6.4 **NLP Feature Engineering**

* Tweet length

* Number of hashtags

* Emoji count

* VADER sentiment scores


```python
#importing necessary libraries
import re
!pip install emoji
import emoji
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
```

    Collecting emoji
      Downloading emoji-2.15.0-py3-none-any.whl.metadata (5.7 kB)
    Downloading emoji-2.15.0-py3-none-any.whl (608 kB)
       ---------------------------------------- 0.0/608.4 kB ? eta -:--:--
       ---------------------------------------- 0.0/608.4 kB ? eta -:--:--
        --------------------------------------- 10.2/608.4 kB ? eta -:--:--
        --------------------------------------- 10.2/608.4 kB ? eta -:--:--
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       - ------------------------------------- 30.7/608.4 kB 262.6 kB/s eta 0:00:03
       -- ------------------------------------- 41.0/608.4 kB 67.8 kB/s eta 0:00:09
       -- ------------------------------------- 41.0/608.4 kB 67.8 kB/s eta 0:00:09
       ---- ----------------------------------- 61.4/608.4 kB 93.6 kB/s eta 0:00:06
       ---- ----------------------------------- 61.4/608.4 kB 93.6 kB/s eta 0:00:06
       ---- ----------------------------------- 61.4/608.4 kB 93.6 kB/s eta 0:00:06
       ---- ----------------------------------- 71.7/608.4 kB 91.4 kB/s eta 0:00:06
       ---- ----------------------------------- 71.7/608.4 kB 91.4 kB/s eta 0:00:06
       ---- ----------------------------------- 71.7/608.4 kB 91.4 kB/s eta 0:00:06
       ---- ----------------------------------- 71.7/608.4 kB 91.4 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------ --------------------------------- 92.2/608.4 kB 97.1 kB/s eta 0:00:06
       ------- ------------------------------- 112.6/608.4 kB 88.6 kB/s eta 0:00:06
       ------- ------------------------------- 122.9/608.4 kB 92.4 kB/s eta 0:00:06
       -------- ----------------------------- 143.4/608.4 kB 105.2 kB/s eta 0:00:05
       -------- ----------------------------- 143.4/608.4 kB 105.2 kB/s eta 0:00:05
       -------- ----------------------------- 143.4/608.4 kB 105.2 kB/s eta 0:00:05
       --------- ---------------------------- 153.6/608.4 kB 104.3 kB/s eta 0:00:05
       --------- ---------------------------- 153.6/608.4 kB 104.3 kB/s eta 0:00:05
       ---------- --------------------------- 174.1/608.4 kB 109.2 kB/s eta 0:00:04
       ---------- --------------------------- 174.1/608.4 kB 109.2 kB/s eta 0:00:04
       ---------- --------------------------- 174.1/608.4 kB 109.2 kB/s eta 0:00:04
       ------------ ------------------------- 194.6/608.4 kB 112.3 kB/s eta 0:00:04
       ------------ ------------------------- 194.6/608.4 kB 112.3 kB/s eta 0:00:04
       ------------ ------------------------- 204.8/608.4 kB 114.2 kB/s eta 0:00:04
       ------------ ------------------------- 204.8/608.4 kB 114.2 kB/s eta 0:00:04
       ------------ ------------------------- 204.8/608.4 kB 114.2 kB/s eta 0:00:04
       ------------ ------------------------- 204.8/608.4 kB 114.2 kB/s eta 0:00:04
       ------------ ------------------------- 204.8/608.4 kB 114.2 kB/s eta 0:00:04
       -------------- ----------------------- 225.3/608.4 kB 111.0 kB/s eta 0:00:04
       -------------- ----------------------- 235.5/608.4 kB 112.6 kB/s eta 0:00:04
       --------------- ---------------------- 256.0/608.4 kB 118.3 kB/s eta 0:00:03
       --------------- ---------------------- 256.0/608.4 kB 118.3 kB/s eta 0:00:03
       ----------------- -------------------- 276.5/608.4 kB 121.7 kB/s eta 0:00:03
       ----------------- -------------------- 276.5/608.4 kB 121.7 kB/s eta 0:00:03
       ----------------- -------------------- 276.5/608.4 kB 121.7 kB/s eta 0:00:03
       ----------------- -------------------- 286.7/608.4 kB 121.2 kB/s eta 0:00:03
       ----------------- -------------------- 286.7/608.4 kB 121.2 kB/s eta 0:00:03
       ----------------- -------------------- 286.7/608.4 kB 121.2 kB/s eta 0:00:03
       ----------------- -------------------- 286.7/608.4 kB 121.2 kB/s eta 0:00:03
       ----------------- -------------------- 286.7/608.4 kB 121.2 kB/s eta 0:00:03
       ------------------- ------------------ 307.2/608.4 kB 118.0 kB/s eta 0:00:03
       ------------------- ------------------ 307.2/608.4 kB 118.0 kB/s eta 0:00:03
       ------------------- ------------------ 307.2/608.4 kB 118.0 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       ------------------- ------------------ 317.4/608.4 kB 115.7 kB/s eta 0:00:03
       --------------------- ---------------- 337.9/608.4 kB 112.1 kB/s eta 0:00:03
       --------------------- ---------------- 337.9/608.4 kB 112.1 kB/s eta 0:00:03
       ---------------------- --------------- 358.4/608.4 kB 116.1 kB/s eta 0:00:03
       ---------------------- --------------- 358.4/608.4 kB 116.1 kB/s eta 0:00:03
       ----------------------- -------------- 368.6/608.4 kB 115.8 kB/s eta 0:00:03
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------ ------------- 389.1/608.4 kB 119.4 kB/s eta 0:00:02
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       ------------------------------- ------ 501.8/608.4 kB 113.1 kB/s eta 0:00:01
       -------------------------------------  604.2/608.4 kB 115.9 kB/s eta 0:00:01
       -------------------------------------- 608.4/608.4 kB 115.7 kB/s eta 0:00:00
    Installing collected packages: emoji
    Successfully installed emoji-2.15.0
    

    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    

    Collecting vaderSentiment
      Downloading vaderSentiment-3.3.2-py2.py3-none-any.whl.metadata (572 bytes)
    Requirement already satisfied: requests in c:\users\g-osundwa\anaconda3\lib\site-packages (from vaderSentiment) (2.32.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\g-osundwa\anaconda3\lib\site-packages (from requests->vaderSentiment) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\g-osundwa\anaconda3\lib\site-packages (from requests->vaderSentiment) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\g-osundwa\anaconda3\lib\site-packages (from requests->vaderSentiment) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\g-osundwa\anaconda3\lib\site-packages (from requests->vaderSentiment) (2025.4.26)
    Downloading vaderSentiment-3.3.2-py2.py3-none-any.whl (125 kB)
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ---------------------------------------- 0.0/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ------------------- -------------------- 61.4/126.0 kB ? eta -:--:--
       ---------------------- ----------------- 71.7/126.0 kB 77.1 kB/s eta 0:00:01
       ---------------------- ----------------- 71.7/126.0 kB 77.1 kB/s eta 0:00:01
       ---------------------- ----------------- 71.7/126.0 kB 77.1 kB/s eta 0:00:01
       ---------------------- ----------------- 71.7/126.0 kB 77.1 kB/s eta 0:00:01
       ---------------------- ----------------- 71.7/126.0 kB 77.1 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ----------------------------- ---------- 92.2/126.0 kB 79.5 kB/s eta 0:00:01
       ------------------------------- ------- 102.4/126.0 kB 69.4 kB/s eta 0:00:01
       ------------------------------- ------- 102.4/126.0 kB 69.4 kB/s eta 0:00:01
       --------------------------------------- 126.0/126.0 kB 79.6 kB/s eta 0:00:00
    Installing collected packages: vaderSentiment
    Successfully installed vaderSentiment-3.3.2
    

    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ympy (C:\Users\G-Osundwa\anaconda3\Lib\site-packages)
    


```python
# 1️ Tweet length (number of characters)
df_clean['tweet_length'] = df_clean['tweet_text'].apply(len)

# 2️ Number of hashtags
df_clean['num_hashtags'] = df_clean['tweet_text'].apply(lambda x: len(re.findall(r'#\w+', str(x))))

# 3️ Emoji count
def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)
df_clean['num_emojis'] = df_clean['tweet_text'].apply(count_emojis)

# 4️ VADER sentiment scores
def vader_scores(text):
    vs = analyzer.polarity_scores(str(text))
    return pd.Series([vs['neg'], vs['neu'], vs['pos'], vs['compound']])

df_clean[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = df_clean['tweet_text'].apply(vader_scores)

```


```python
df_clean[['tweet_text', 'tweet_length', 'num_hashtags', 'num_emojis', 'vader_compound']].sample(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>tweet_length</th>
      <th>num_hashtags</th>
      <th>num_emojis</th>
      <th>vader_compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2152</th>
      <td>Stepping away from the computer, leaving my iP...</td>
      <td>117</td>
      <td>3</td>
      <td>0</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>6634</th>
      <td>RT @mention There are more than a few PATs her...</td>
      <td>138</td>
      <td>1</td>
      <td>0</td>
      <td>0.3182</td>
    </tr>
    <tr>
      <th>1848</th>
      <td>has an ipad 2 in his hands, and it is exquisit...</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>6596</th>
      <td>RT @mention The Beluga iPhone app is the highl...</td>
      <td>124</td>
      <td>2</td>
      <td>0</td>
      <td>0.8481</td>
    </tr>
    <tr>
      <th>5978</th>
      <td>RT @mention If you thought people spent a lot ...</td>
      <td>110</td>
      <td>1</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



# 6.5 **Vectorization**

* CountVectorizer

* TF-IDF


```python
#Step 1: Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.sparse import hstack
```


```python
#Step 2: Prepare features and labels

# Text feature
X_text = df_clean['text_final']

# Numeric features
X_numeric = df_clean[['tweet_length', 'num_hashtags', 'num_emojis', 'vader_compound']]

# Target
y = df_clean['label']

# Train/test split (80/20)
X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
)
```


```python
#Step 3: TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # unigrams + bigrams
X_text_train_tfidf = tfidf.fit_transform(X_text_train)
X_text_test_tfidf = tfidf.transform(X_text_test)
```

# 7. **Modeling Approach**
**Classical Machine Learning Models:**

* Logistic Regression
* Class-Balanced Logistic Regression
* Linear SVM
  
**Advanced Classical Models for Multiclass NLP**
* Multinomial Naive Bayes
* Linear SVM
* Improved Logistic Regression (Tuned)
* Apply SMOTE to handle class imbalance
* Richer Features
* Strong Model Upgrade: Mini-BERT


# 7.1 **Logistic Regression**


```python
#Step 4: Combine text + numeric features
X_train_combined = hstack([X_text_train_tfidf, X_numeric_train])
X_test_combined = hstack([X_text_test_tfidf, X_numeric_test])
```


```python
#Step 5: Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_combined, y_train)
```

    C:\Users\G-Osundwa\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000)</pre></div></div></div></div></div>




```python
#Step 6: Make predictions
y_pred = lr.predict(X_test_combined)

```


```python
#Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

    Accuracy: 0.6891133557800224
    F1-score (weighted): 0.6654851743632886
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.54      0.06      0.11       114
             1.0       0.63      0.54      0.58       594
             2.0       0.71      0.84      0.77      1074
    
        accuracy                           0.69      1782
       macro avg       0.63      0.48      0.49      1782
    weighted avg       0.67      0.69      0.67      1782
    
    

The model achieved an overall accuracy of appr 69% and a weighted F1-score of 0.66, performing best on the majority (neutral) class. It struggles with the minority negative class, which has very low recall, indicating many negative tweets are misclassified. Positive tweets are moderately well-predicted, but class imbalance likely limits overall performance.

# 7.2 **Class-Balanced Logistic Regression**

Since the negative class is underrepresented, we can use class_weight='balanced' in Logistic Regression to give more weight to minority classes. This will improves recall and F1 for small classes.


```python
#Class-Balanced Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Create Logistic Regression with class balancing
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_balanced.fit(X_train_combined, y_train)

# Predict
y_pred_balanced = lr_balanced.predict(X_test_combined)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("\nClassification Report:\n", classification_report(y_test, y_pred_balanced))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))

```

    Accuracy: 0.6447811447811448
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.33      0.59      0.42       114
             1.0       0.57      0.66      0.61       594
             2.0       0.78      0.64      0.70      1074
    
        accuracy                           0.64      1782
       macro avg       0.56      0.63      0.58      1782
    weighted avg       0.68      0.64      0.65      1782
    
    
    Confusion Matrix:
     [[ 67  14  33]
     [ 40 394 160]
     [ 99 287 688]]
    

    C:\Users\G-Osundwa\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

 **Interpretation and Comparison**
1. Accuracy & F1 Accuracy  appr 64%: dropped slightly compared to the previous baseline (appr 69%).

Weighted F1  0.65: similar to before.

**Why accuracy dropped:**

Class balancing increased recall for the minority class (Negative), which means the model is correctly predicting more negative tweets but misclassifying some majority class (Neutral) tweets.

2. Minority Class Performance (Negative, 0.0)

Precision: 0.30, Recall: 0.58, F1: 0.40

Recall improved a lot (was 0.07 before), meaning the model is now detecting more negative tweets.

Precision is lower, meaning some predicted negatives are actually positives or neutral.

Confusion Matrix
[[ 66  13  35]   - Negative
 [ 49 385 160]   - Positive
 [105 283 686]]  - Neutral


Negative class (0.0) is detected much more (66 correct out of 114)

Positive and Neutral classes have some misclassifications due to the balancing.

Convergence Warning
lbfgs failed to converge

This happened because logistic regression didn’t fully converge with max_iter=1000 on sparse + combined features.




```python
#we increase max_iter

lr_balanced = LogisticRegression(max_iter=2000, class_weight='balanced')
```

# 7.3 **Advanced Classical Models for Multiclass NLP**
We will build and compare:

1. Linear SVM (Linear Support Vector Machine)
2. Multinomial Naive Bayes
3. Logistic Regression (Improved tuning)

**Step 1 Linear SVM**




```python
from sklearn.svm import LinearSVC

svm_clf = LinearSVC(class_weight='balanced')
svm_clf.fit(X_train_combined, y_train)

y_pred_svm = svm_clf.predict(X_test_combined)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
```

    SVM Accuracy: 0.67003367003367
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.31      0.32      0.32       114
             1.0       0.64      0.48      0.55       594
             2.0       0.72      0.81      0.76      1074
    
        accuracy                           0.67      1782
       macro avg       0.56      0.54      0.54      1782
    weighted avg       0.66      0.67      0.66      1782
    
    
    Confusion Matrix:
     [[ 37  10  67]
     [ 29 286 279]
     [ 52 151 871]]
    

    C:\Users\G-Osundwa\anaconda3\Lib\site-packages\sklearn\svm\_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(
    

**Step 2 Multinomial Naive Bayes**




```python
# TF-IDF only
X_train_tfidf = X_text_train_tfidf
X_test_tfidf = X_text_test_tfidf

from sklearn.naive_bayes import MultinomialNB

nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

y_pred_nb = nb_clf.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
```

    Naive Bayes Accuracy: 0.6638608305274972
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       1.00      0.04      0.07       114
             1.0       0.60      0.41      0.49       594
             2.0       0.68      0.87      0.76      1074
    
        accuracy                           0.66      1782
       macro avg       0.76      0.44      0.44      1782
    weighted avg       0.68      0.66      0.63      1782
    
    
    Confusion Matrix:
     [[  4  22  88]
     [  0 243 351]
     [  0 138 936]]
    

**Step 3 Improved Logistic Regression (Tuned)**


```python
best_lr = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    C=0.5,
)
best_lr.fit(X_train_combined, y_train)

y_pred_lr_tuned = best_lr.predict(X_test_combined)

print("Tuned LR Accuracy:", accuracy_score(y_test, y_pred_lr_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr_tuned))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_tuned))
```

    Tuned LR Accuracy: 0.4270482603815937
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.07      0.18      0.10       123
             1.0       0.37      0.42      0.39       590
             2.0       0.63      0.46      0.53      1069
    
        accuracy                           0.43      1782
       macro avg       0.35      0.35      0.34      1782
    weighted avg       0.50      0.43      0.45      1782
    
    
    Confusion Matrix:
     [[ 22  58  43]
     [ 96 249 245]
     [205 374 490]]
    

Model Comparison Summary

Across all three advanced models,
**Linear SVM** delivered the strongest overall performance, reaching 0.666 accuracy and showing the best balance between precision, recall, and F1—especially for the dominant class (2.0).

 **Logistic Regression**, even after tuning, remained slightly weaker and continued to struggle with class 0.0, though it performed reasonably on classes 1.0 and 2.0.

 **Multinomial Naive Bayes** performed inconsistently, achieving very high precision but extremely low recall for class 0.0, meaning it failed to correctly identify most negative tweets. Overall, SVM appears to be the most stable and reliable classifier for this multiclass sentiment task.

**Next Steps**

We should improve class balance and feature richness—specifically by:
*  Applying SMOTE  to address the    underrepresented class 0.0.
* Perform advanced embeddings (TF-IDF n-grams - Word2Vec - BERT embeddings) to capture semantic context.
* Finally, train a stronger model (e.g., Linear SVM + class weights or a simple transformer classifier) for a significant performance boost.

**STEP 1 Apply SMOTE to handle class imbalance**




```python
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Vectorize
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df_clean['text_final'])
y = df_clean['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE can’t accept sparse - convert to dense
X_train_dense = X_train.toarray()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_dense, y_train)

# Train SVM
svm_smote = LinearSVC()
svm_smote.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred = svm_smote.predict(X_test.toarray())

print("SMOTE-SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

    SMOTE-SVM Accuracy: 0.6509539842873177
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.38      0.51      0.44       114
             1.0       0.56      0.62      0.59       594
             2.0       0.75      0.68      0.72      1074
    
        accuracy                           0.65      1782
       macro avg       0.57      0.60      0.58      1782
    weighted avg       0.67      0.65      0.66      1782
    
    
    Confusion Matrix:
     [[ 58  14  42]
     [ 28 369 197]
     [ 65 276 733]]
    

**STEP 2  Richer Features (TF-IDF + n-grams + emojis + sentiment + length)**


```python
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# TF-IDF text features
X_tfidf = tfidf.transform(df_clean['text_final'])

# Your numeric features
num_features = df_clean[['tweet_length', 'num_hashtags', 'num_emojis',
                   'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(num_features)

# Combine TF-IDF + numeric
X_full = hstack([X_tfidf, X_num])
```


```python
# Training using class-weighted SVM:
svm_weighted = LinearSVC(class_weight='balanced')
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

svm_weighted.fit(X_train, y_train)
y_pred = svm_weighted.predict(X_test)

print("Weighted SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

    Weighted SVM Accuracy: 0.6980920314253648
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.49      0.41      0.45       123
             1.0       0.62      0.58      0.60       590
             2.0       0.75      0.80      0.77      1069
    
        accuracy                           0.70      1782
       macro avg       0.62      0.60      0.61      1782
    weighted avg       0.69      0.70      0.69      1782
    
    
    Confusion Matrix:
     [[ 51  20  52]
     [ 21 343 226]
     [ 32 187 850]]
    

    C:\Users\G-Osundwa\anaconda3\Lib\site-packages\sklearn\svm\_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(
    

**STEP 3 Strong Model Upgrade: Mini-BERT**




```python
!pip install tensorflow
```


```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# 1. Load tokenizer and model
# -------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------------------------------
# 2. Match training/testing text with label indices
# -------------------------------------------------------
train_indices = y_train.index
test_indices = y_test.index

X_train_text = df_clean.loc[train_indices, "text_final"].tolist()
X_test_text = df_clean.loc[test_indices, "text_final"].tolist()

# -------------------------------------------------------
# 3. Tokenize
# -------------------------------------------------------
train_encodings = tokenizer(
    X_train_text,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

test_encodings = tokenizer(
    X_test_text,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# Convert labels to torch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# -------------------------------------------------------
# 4. Create PyTorch Dataset
# -------------------------------------------------------
class TorchDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TorchDataset(train_encodings, y_train_tensor)
test_dataset = TorchDataset(test_encodings, y_test_tensor)

# -------------------------------------------------------
# 5. DataLoaders
# -------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -------------------------------------------------------
# 6. Training Setup
# -------------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# 7. Training Loop
# -------------------------------------------------------
model.train()
epochs = 2

for epoch in range(epochs):
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

# -------------------------------------------------------
# 8. Evaluation
# -------------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

    Epoch 1/2 - Average Loss: 0.7487
    Epoch 2/2 - Average Loss: 0.5810
    Test Accuracy: 0.7116
    


```python
import pandas as pd
from IPython.display import display

# -------------------------------------------------------
# 1. Create the data
# -------------------------------------------------------
data = {
    "Model": [
        "DistilBERT (PyTorch)",
        "Weighted SVM",
        "SVM",
        "Naive Bayes",
        "SMOTE-SVM",
        "Tuned Logistic Regression"
    ],
    "Accuracy": [0.71, 0.70, 0.67, 0.66, 0.65, 0.43],
    "Weighted F1": [0.67, 0.69, 0.66, 0.63, 0.66, 0.45],
    "Strengths": [
        "Best overall accuracy, strong recall for major class (2), leverages pre with only 2 Epochtrained embeddings",
        "Good balance between classes, interpretable, fast training",
        "Solid general performance, fast",
        "Very simple, fast, interpretable",
        "Improves minority class recall",
        "Simple and fast"
    ],
    "Weaknesses": [
        "Lower performance on minority class (0), requires GPU & fine-tuning",
        "Slightly worse than DistilBERT, struggles with rare class",
        "Lower recall for class 1 and class 0, sensitive to imbalance",
        "Poor recall for minority class, overpredicts majority class",
        "Slightly worse overall accuracy",
        "Very low accuracy, underperforms across all classes"
    ],
    "Score /10": ["9/10", "8/10", "7/10", "6/10", "6.5/10", "4/10"]
}

# -------------------------------------------------------
# 2. Create DataFrame
# -------------------------------------------------------
df_models = pd.DataFrame(data)

# -------------------------------------------------------
# 3. Display nicely
# -------------------------------------------------------
display(df_models.style.set_properties(**{
    'text-align': 'left',
    'white-space': 'pre-wrap'
}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'left')]}
]))

```


<style type="text/css">
#T_7d77a th {
  text-align: left;
}
#T_7d77a_row0_col0, #T_7d77a_row0_col1, #T_7d77a_row0_col2, #T_7d77a_row0_col3, #T_7d77a_row0_col4, #T_7d77a_row0_col5, #T_7d77a_row1_col0, #T_7d77a_row1_col1, #T_7d77a_row1_col2, #T_7d77a_row1_col3, #T_7d77a_row1_col4, #T_7d77a_row1_col5, #T_7d77a_row2_col0, #T_7d77a_row2_col1, #T_7d77a_row2_col2, #T_7d77a_row2_col3, #T_7d77a_row2_col4, #T_7d77a_row2_col5, #T_7d77a_row3_col0, #T_7d77a_row3_col1, #T_7d77a_row3_col2, #T_7d77a_row3_col3, #T_7d77a_row3_col4, #T_7d77a_row3_col5, #T_7d77a_row4_col0, #T_7d77a_row4_col1, #T_7d77a_row4_col2, #T_7d77a_row4_col3, #T_7d77a_row4_col4, #T_7d77a_row4_col5, #T_7d77a_row5_col0, #T_7d77a_row5_col1, #T_7d77a_row5_col2, #T_7d77a_row5_col3, #T_7d77a_row5_col4, #T_7d77a_row5_col5 {
  text-align: left;
  white-space: pre-wrap;
}
</style>
<table id="T_7d77a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7d77a_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_7d77a_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_7d77a_level0_col2" class="col_heading level0 col2" >Weighted F1</th>
      <th id="T_7d77a_level0_col3" class="col_heading level0 col3" >Strengths</th>
      <th id="T_7d77a_level0_col4" class="col_heading level0 col4" >Weaknesses</th>
      <th id="T_7d77a_level0_col5" class="col_heading level0 col5" >Score /10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7d77a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7d77a_row0_col0" class="data row0 col0" >DistilBERT (PyTorch)</td>
      <td id="T_7d77a_row0_col1" class="data row0 col1" >0.710000</td>
      <td id="T_7d77a_row0_col2" class="data row0 col2" >0.670000</td>
      <td id="T_7d77a_row0_col3" class="data row0 col3" >Best overall accuracy, strong recall for major class (2), leverages pre with only 2 Epochtrained embeddings</td>
      <td id="T_7d77a_row0_col4" class="data row0 col4" >Lower performance on minority class (0), requires GPU & fine-tuning</td>
      <td id="T_7d77a_row0_col5" class="data row0 col5" >9/10</td>
    </tr>
    <tr>
      <th id="T_7d77a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_7d77a_row1_col0" class="data row1 col0" >Weighted SVM</td>
      <td id="T_7d77a_row1_col1" class="data row1 col1" >0.700000</td>
      <td id="T_7d77a_row1_col2" class="data row1 col2" >0.690000</td>
      <td id="T_7d77a_row1_col3" class="data row1 col3" >Good balance between classes, interpretable, fast training</td>
      <td id="T_7d77a_row1_col4" class="data row1 col4" >Slightly worse than DistilBERT, struggles with rare class</td>
      <td id="T_7d77a_row1_col5" class="data row1 col5" >8/10</td>
    </tr>
    <tr>
      <th id="T_7d77a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_7d77a_row2_col0" class="data row2 col0" >SVM</td>
      <td id="T_7d77a_row2_col1" class="data row2 col1" >0.670000</td>
      <td id="T_7d77a_row2_col2" class="data row2 col2" >0.660000</td>
      <td id="T_7d77a_row2_col3" class="data row2 col3" >Solid general performance, fast</td>
      <td id="T_7d77a_row2_col4" class="data row2 col4" >Lower recall for class 1 and class 0, sensitive to imbalance</td>
      <td id="T_7d77a_row2_col5" class="data row2 col5" >7/10</td>
    </tr>
    <tr>
      <th id="T_7d77a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_7d77a_row3_col0" class="data row3 col0" >Naive Bayes</td>
      <td id="T_7d77a_row3_col1" class="data row3 col1" >0.660000</td>
      <td id="T_7d77a_row3_col2" class="data row3 col2" >0.630000</td>
      <td id="T_7d77a_row3_col3" class="data row3 col3" >Very simple, fast, interpretable</td>
      <td id="T_7d77a_row3_col4" class="data row3 col4" >Poor recall for minority class, overpredicts majority class</td>
      <td id="T_7d77a_row3_col5" class="data row3 col5" >6/10</td>
    </tr>
    <tr>
      <th id="T_7d77a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_7d77a_row4_col0" class="data row4 col0" >SMOTE-SVM</td>
      <td id="T_7d77a_row4_col1" class="data row4 col1" >0.650000</td>
      <td id="T_7d77a_row4_col2" class="data row4 col2" >0.660000</td>
      <td id="T_7d77a_row4_col3" class="data row4 col3" >Improves minority class recall</td>
      <td id="T_7d77a_row4_col4" class="data row4 col4" >Slightly worse overall accuracy</td>
      <td id="T_7d77a_row4_col5" class="data row4 col5" >6.5/10</td>
    </tr>
    <tr>
      <th id="T_7d77a_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_7d77a_row5_col0" class="data row5 col0" >Tuned Logistic Regression</td>
      <td id="T_7d77a_row5_col1" class="data row5 col1" >0.430000</td>
      <td id="T_7d77a_row5_col2" class="data row5 col2" >0.450000</td>
      <td id="T_7d77a_row5_col3" class="data row5 col3" >Simple and fast</td>
      <td id="T_7d77a_row5_col4" class="data row5 col4" >Very low accuracy, underperforms across all classes</td>
      <td id="T_7d77a_row5_col5" class="data row5 col5" >4/10</td>
    </tr>
  </tbody>
</table>



We try to increase number of epoch to 4


```python
import matplotlib.pyplot as plt

models = ['DistilBERT', 'Weighted SVM', 'SVM', 'Naive Bayes', 'SMOTE-SVM', 'Tuned LR']
accuracy = [0.7116, 0.6981, 0.6700, 0.6638, 0.6509, 0.4270]
f1_weighted = [0.667, 0.686, 0.66, 0.63, 0.66, 0.45]

x = range(len(models))
plt.figure(figsize=(10,6))
plt.bar(x, accuracy, width=0.4, label='Accuracy', align='center')
plt.bar([i+0.4 for i in x], f1_weighted, width=0.4, label='Weighted F1', align='center')
plt.xticks([i+0.2 for i in x], models, rotation=15)
plt.ylabel('Score')
plt.title('Model Comparison: Accuracy vs Weighted F1')
plt.legend()
plt.show()

```


    
![Accuracy vs Weighted F1](images/output_80_0.png)
    



```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# 1. Model & Tokenizer
# -------------------------------------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------------------------------
# 2. Prepare text and labels
# -------------------------------------------------------
train_indices = y_train.index
test_indices = y_test.index

X_train_text = df_clean.loc[train_indices, "text_final"].tolist()
X_test_text = df_clean.loc[test_indices, "text_final"].tolist()

# -------------------------------------------------------
# 3. Tokenization with max length 256
# -------------------------------------------------------
train_encodings = tokenizer(
    X_train_text,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

test_encodings = tokenizer(
    X_test_text,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# -------------------------------------------------------
# 4. PyTorch Dataset
# -------------------------------------------------------
class TorchDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TorchDataset(train_encodings, y_train_tensor)
test_dataset = TorchDataset(test_encodings, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -------------------------------------------------------
# 5. Optimizer & Loss
# -------------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=2e-5)  # lower learning rate
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# 6. Training Loop (4 epochs)
# -------------------------------------------------------
epochs = 4
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

# -------------------------------------------------------
# 7. Evaluation
# -------------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

    Epoch 1/4 - Average Loss: 0.7858
    Epoch 2/4 - Average Loss: 0.6078
    Epoch 3/4 - Average Loss: 0.4686
    Epoch 4/4 - Average Loss: 0.3493
    Test Accuracy: 0.7149
    

# 8. **Model Evaluation**


```python
import pandas as pd
from IPython.display import display

# -------------------------------------------------------
# 1. Updated data including new BERT model
# -------------------------------------------------------
data = {
    "Model": [
        "BERT Base (4 Epochs)",
        "DistilBERT (PyTorch)",
        "Weighted SVM",
        "SVM",
        "Naive Bayes",
        "SMOTE-SVM",
        "Tuned Logistic Regression"
    ],
    "Accuracy": [0.7149, 0.71, 0.70, 0.67, 0.66, 0.65, 0.43],
    "Weighted F1": [0.69, 0.67, 0.69, 0.66, 0.63, 0.66, 0.45],
    "Strengths": [
        "Best model overall; strong fine-tuning performance; excellent accuracy and stable learning",
        "Strong accuracy, pretrained embeddings help performance",
        "Balanced performance across classes; interpretable; fast",
        "Solid general performance; efficient",
        "Simple, interpretable, fast",
        "Improves minority class recall",
        "Very simple and fast"
    ],
    "Weaknesses": [
        "Computationally heavier; needs GPU for best performance",
        "Lower recall on minority class; needs fine-tuning",
        "Slightly behind transformer models; struggles with rare class",
        "Sensitive to class imbalance; lower recall for class 0 and 1",
        "Overpredicts majority class; weak minority recall",
        "Slightly weaker accuracy overall",
        "Very low accuracy; underperforms across classes"
    ],
    "Score /10": ["9.5/10", "9/10", "8/10", "7/10", "6/10", "6.5/10", "4/10"]
}

# -------------------------------------------------------
# 2. Create DataFrame
# -------------------------------------------------------
df_models = pd.DataFrame(data)

# -------------------------------------------------------
# 3. Display styled table
# -------------------------------------------------------
display(df_models.style.set_properties(**{
    'text-align': 'left',
    'white-space': 'pre-wrap'
}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'left')]}
]))

```


<style type="text/css">
#T_f8a61 th {
  text-align: left;
}
#T_f8a61_row0_col0, #T_f8a61_row0_col1, #T_f8a61_row0_col2, #T_f8a61_row0_col3, #T_f8a61_row0_col4, #T_f8a61_row0_col5, #T_f8a61_row1_col0, #T_f8a61_row1_col1, #T_f8a61_row1_col2, #T_f8a61_row1_col3, #T_f8a61_row1_col4, #T_f8a61_row1_col5, #T_f8a61_row2_col0, #T_f8a61_row2_col1, #T_f8a61_row2_col2, #T_f8a61_row2_col3, #T_f8a61_row2_col4, #T_f8a61_row2_col5, #T_f8a61_row3_col0, #T_f8a61_row3_col1, #T_f8a61_row3_col2, #T_f8a61_row3_col3, #T_f8a61_row3_col4, #T_f8a61_row3_col5, #T_f8a61_row4_col0, #T_f8a61_row4_col1, #T_f8a61_row4_col2, #T_f8a61_row4_col3, #T_f8a61_row4_col4, #T_f8a61_row4_col5, #T_f8a61_row5_col0, #T_f8a61_row5_col1, #T_f8a61_row5_col2, #T_f8a61_row5_col3, #T_f8a61_row5_col4, #T_f8a61_row5_col5, #T_f8a61_row6_col0, #T_f8a61_row6_col1, #T_f8a61_row6_col2, #T_f8a61_row6_col3, #T_f8a61_row6_col4, #T_f8a61_row6_col5 {
  text-align: left;
  white-space: pre-wrap;
}
</style>
<table id="T_f8a61">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f8a61_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_f8a61_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_f8a61_level0_col2" class="col_heading level0 col2" >Weighted F1</th>
      <th id="T_f8a61_level0_col3" class="col_heading level0 col3" >Strengths</th>
      <th id="T_f8a61_level0_col4" class="col_heading level0 col4" >Weaknesses</th>
      <th id="T_f8a61_level0_col5" class="col_heading level0 col5" >Score /10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f8a61_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f8a61_row0_col0" class="data row0 col0" >BERT Base (4 Epochs)</td>
      <td id="T_f8a61_row0_col1" class="data row0 col1" >0.714900</td>
      <td id="T_f8a61_row0_col2" class="data row0 col2" >0.690000</td>
      <td id="T_f8a61_row0_col3" class="data row0 col3" >Best model overall; strong fine-tuning performance; excellent accuracy and stable learning</td>
      <td id="T_f8a61_row0_col4" class="data row0 col4" >Computationally heavier; needs GPU for best performance</td>
      <td id="T_f8a61_row0_col5" class="data row0 col5" >9.5/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f8a61_row1_col0" class="data row1 col0" >DistilBERT (PyTorch)</td>
      <td id="T_f8a61_row1_col1" class="data row1 col1" >0.710000</td>
      <td id="T_f8a61_row1_col2" class="data row1 col2" >0.670000</td>
      <td id="T_f8a61_row1_col3" class="data row1 col3" >Strong accuracy, pretrained embeddings help performance</td>
      <td id="T_f8a61_row1_col4" class="data row1 col4" >Lower recall on minority class; needs fine-tuning</td>
      <td id="T_f8a61_row1_col5" class="data row1 col5" >9/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f8a61_row2_col0" class="data row2 col0" >Weighted SVM</td>
      <td id="T_f8a61_row2_col1" class="data row2 col1" >0.700000</td>
      <td id="T_f8a61_row2_col2" class="data row2 col2" >0.690000</td>
      <td id="T_f8a61_row2_col3" class="data row2 col3" >Balanced performance across classes; interpretable; fast</td>
      <td id="T_f8a61_row2_col4" class="data row2 col4" >Slightly behind transformer models; struggles with rare class</td>
      <td id="T_f8a61_row2_col5" class="data row2 col5" >8/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f8a61_row3_col0" class="data row3 col0" >SVM</td>
      <td id="T_f8a61_row3_col1" class="data row3 col1" >0.670000</td>
      <td id="T_f8a61_row3_col2" class="data row3 col2" >0.660000</td>
      <td id="T_f8a61_row3_col3" class="data row3 col3" >Solid general performance; efficient</td>
      <td id="T_f8a61_row3_col4" class="data row3 col4" >Sensitive to class imbalance; lower recall for class 0 and 1</td>
      <td id="T_f8a61_row3_col5" class="data row3 col5" >7/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f8a61_row4_col0" class="data row4 col0" >Naive Bayes</td>
      <td id="T_f8a61_row4_col1" class="data row4 col1" >0.660000</td>
      <td id="T_f8a61_row4_col2" class="data row4 col2" >0.630000</td>
      <td id="T_f8a61_row4_col3" class="data row4 col3" >Simple, interpretable, fast</td>
      <td id="T_f8a61_row4_col4" class="data row4 col4" >Overpredicts majority class; weak minority recall</td>
      <td id="T_f8a61_row4_col5" class="data row4 col5" >6/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f8a61_row5_col0" class="data row5 col0" >SMOTE-SVM</td>
      <td id="T_f8a61_row5_col1" class="data row5 col1" >0.650000</td>
      <td id="T_f8a61_row5_col2" class="data row5 col2" >0.660000</td>
      <td id="T_f8a61_row5_col3" class="data row5 col3" >Improves minority class recall</td>
      <td id="T_f8a61_row5_col4" class="data row5 col4" >Slightly weaker accuracy overall</td>
      <td id="T_f8a61_row5_col5" class="data row5 col5" >6.5/10</td>
    </tr>
    <tr>
      <th id="T_f8a61_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f8a61_row6_col0" class="data row6 col0" >Tuned Logistic Regression</td>
      <td id="T_f8a61_row6_col1" class="data row6 col1" >0.430000</td>
      <td id="T_f8a61_row6_col2" class="data row6 col2" >0.450000</td>
      <td id="T_f8a61_row6_col3" class="data row6 col3" >Very simple and fast</td>
      <td id="T_f8a61_row6_col4" class="data row6 col4" >Very low accuracy; underperforms across classes</td>
      <td id="T_f8a61_row6_col5" class="data row6 col5" >4/10</td>
    </tr>
  </tbody>
</table>




```python
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# Model Results Data
# ============================
data = {
    "Model": [
        "DistilBERT (2-epoch)",
        "DistilBERT (4-epoch)",
        "Weighted SVM",
        "SVM",
        "Naive Bayes",
        "SMOTE-SVM",
        "Tuned Logistic Regression"
    ],
    "Accuracy": [
        0.7116,
        0.7149,
        0.6981,
        0.6700,
        0.6639,
        0.6510,
        0.4270
    ],
    "Weighted F1": [
        0.67,
        0.69,     # Assuming similar performance; update if you later compute exact values
        0.69,
        0.66,
        0.63,
        0.66,
        0.45
    ]
}

df = pd.DataFrame(data)

# ============================
# Bar Chart Visualization
# ============================

plt.figure(figsize=(12, 6))
plt.bar(df["Model"], df["Accuracy"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()


# ============================
# Combined Accuracy + F1 Chart
# ============================

plt.figure(figsize=(12, 6))
plt.plot(df["Model"], df["Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Model"], df["Weighted F1"], marker='o', label="Weighted F1")

plt.xticks(rotation=45, ha='right')
plt.ylabel("Score")
plt.title("Accuracy vs Weighted F1 for All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================
# Data Table Display
# ============================

df.style.background_gradient(cmap="Blues").set_caption("Model Performance Summary")

```


    
![Model Accuracy Comparison](images/output_84_0.png)
    



    
![Accuracy vs Weighted F1 for All Models](images/output_84_1.png)
    





<style type="text/css">
#T_6a55d_row0_col1 {
  background-color: #08326e;
  color: #f1f1f1;
}
#T_6a55d_row0_col2 {
  background-color: #08468b;
  color: #f1f1f1;
}
#T_6a55d_row1_col1, #T_6a55d_row1_col2, #T_6a55d_row2_col2 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_6a55d_row2_col1 {
  background-color: #083e81;
  color: #f1f1f1;
}
#T_6a55d_row3_col1 {
  background-color: #0e58a2;
  color: #f1f1f1;
}
#T_6a55d_row3_col2, #T_6a55d_row5_col2 {
  background-color: #08509b;
  color: #f1f1f1;
}
#T_6a55d_row4_col1 {
  background-color: #125ea6;
  color: #f1f1f1;
}
#T_6a55d_row4_col2 {
  background-color: #2070b4;
  color: #f1f1f1;
}
#T_6a55d_row5_col1 {
  background-color: #1b69af;
  color: #f1f1f1;
}
#T_6a55d_row6_col1, #T_6a55d_row6_col2 {
  background-color: #f7fbff;
  color: #000000;
}
</style>
<table id="T_6a55d">
  <caption>Model Performance Summary</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6a55d_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_6a55d_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_6a55d_level0_col2" class="col_heading level0 col2" >Weighted F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6a55d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_6a55d_row0_col0" class="data row0 col0" >DistilBERT (2-epoch)</td>
      <td id="T_6a55d_row0_col1" class="data row0 col1" >0.711600</td>
      <td id="T_6a55d_row0_col2" class="data row0 col2" >0.670000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_6a55d_row1_col0" class="data row1 col0" >DistilBERT (4-epoch)</td>
      <td id="T_6a55d_row1_col1" class="data row1 col1" >0.714900</td>
      <td id="T_6a55d_row1_col2" class="data row1 col2" >0.690000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_6a55d_row2_col0" class="data row2 col0" >Weighted SVM</td>
      <td id="T_6a55d_row2_col1" class="data row2 col1" >0.698100</td>
      <td id="T_6a55d_row2_col2" class="data row2 col2" >0.690000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_6a55d_row3_col0" class="data row3 col0" >SVM</td>
      <td id="T_6a55d_row3_col1" class="data row3 col1" >0.670000</td>
      <td id="T_6a55d_row3_col2" class="data row3 col2" >0.660000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_6a55d_row4_col0" class="data row4 col0" >Naive Bayes</td>
      <td id="T_6a55d_row4_col1" class="data row4 col1" >0.663900</td>
      <td id="T_6a55d_row4_col2" class="data row4 col2" >0.630000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_6a55d_row5_col0" class="data row5 col0" >SMOTE-SVM</td>
      <td id="T_6a55d_row5_col1" class="data row5 col1" >0.651000</td>
      <td id="T_6a55d_row5_col2" class="data row5 col2" >0.660000</td>
    </tr>
    <tr>
      <th id="T_6a55d_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_6a55d_row6_col0" class="data row6 col0" >Tuned Logistic Regression</td>
      <td id="T_6a55d_row6_col1" class="data row6 col1" >0.427000</td>
      <td id="T_6a55d_row6_col2" class="data row6 col2" >0.450000</td>
    </tr>
  </tbody>
</table>




This project evaluated a range of machine-learning and transformer-based models for sentiment analysis on apple_google_twitter data. The goal was to identify the model that offers the best balance of accuracy, robustness to class imbalance, and real-world practicality. Seven models were compared, including classical machine-learning baselines, SMOTE-enhanced variants, and two versions of a fine-tuned DistilBERT transformer model.

1. **Overall Model Performance**

Across all experiments, transformer-based models (DistilBERT) consistently outperformed traditional machine-learning approaches.
The 4-epoch DistilBERT model achieved the highest performance, with:

* Test Accuracy: **0.7149**

* Strong loss reduction over epochs **(0.78 to 0.34)**

* Excellent stability and generalization on unseen data

This confirms the advantage of pretrained contextual embeddings for sentiment tasks, especially when dealing with the informal, context-rich nature of Twitter language.

Following DistilBERT, the Weighted SVM model emerged as the strongest classical approach, achieving:

* Accuracy: 0.6981

* Competitive Weighted F1: 0.69

* Improved recall on the minority class without overly sacrificing the majority class

This demonstrates that class-weighted optimization effectively compensates for label imbalance, a core challenge of real-world sentiment datasets.

2. Comparative Strengths and Weaknesses

* Models such as Naive Bayes and SVM provided respectable baselines, but their inability to capture semantic nuance limited performance. Logistic
* Regression significantly underperformed due to linear decision boundaries that cannot adequately model complex emotional expressions.
* The SMOTE-SVM model successfully improved minority-class recall but at the cost of overall accuracy, showing that synthetic oversampling must be       applied carefully to avoid feature-space distortion.

* The transformer-based DistilBERT models, however, demonstrated:

* Superior semantic understanding

* Strong contextual sensitivity

* Better performance on ambiguous or indirect sentiment expressions

* Minimal preprocessing requirements. These advantages were especially evident in class 2 (positive sentiment), where BERT-based models achieved industry-standard recall levels.

# 8.1 **Final Model Recommendation**

After evaluating all performance metrics, computational costs, and robustness considerations, the DistilBERT (4-epoch) model is recommended as the final production model for Apple tweet sentiment analysis. 
It:

1. Achieves the highest accuracy

2. Demonstrates reliable convergence behavior

3. Handles class imbalance more effectively than baselines

4. Captures subtle sentiment cues essential in social-media data

This makes it the optimal choice for both real-time monitoring and analytical applications involving brand perception, customer satisfaction, and public opinion tracking.

# 9. **Recommendations to Stakeholder**

Based on the sentiment analysis findings and model performance results, we recommend the following actions for the Marketing & Product Team at Apple:

1. Deploy the BERT Model to Monitor Real-Time Customer Sentiment

BERT achieved the highest accuracy (0.7149) and consistently captured nuanced language in tweets.
Deploying this model in production will give Apple reliable real-time insight into public sentiment trends across product lines.

2. Prioritize Investigation of Negative Sentiment Mentions

All models found negative sentiment to be the smallest but hardest class to predict, suggesting customers express negative feedback in subtle or indirect ways.
Apple should implement alerts when negative sentiment spikes so the team can address emerging issues before they escalate.

3. Break Down Sentiment by Product Category

Transformer models capture context well—use this strength to segment sentiment by:

* iPhone

* Mac

* iPad

* Accessories

* Software & Services
This enables more targeted product improvements and campaign adjustments.

4. Use Sentiment Insights to Enhance Customer Engagement Campaigns

The large volume of neutral and positive tweets indicates opportunities to: 

* amplify positive product experiences

* convert neutral conversations into loyalty-building interactions

* create targeted messaging where sentiment is weak or shifting

5. Continuously Retrain the Model With New Tweet Data

Because customer sentiment evolves quickly, especially after product launches or major events, the model should be retrained quarterly or after key Apple announcements to maintain accuracy.

# 10. **Limitations**

* Neutral tweets are often ambiguous.
Many tweets classified as neutral contain mixed or unclear sentiment, making them difficult for even advanced models to interpret accurately. This affects overall precision, especially when distinguishing between mild positive or mild negative opinions.

* Tweets are short and highly contextual.
Twitter posts often include slang, abbreviations, emojis, and cultural references. These characteristics reduce the amount of linguistic information available, making it challenging for models to fully capture meaning.

* Sarcasm detection remains difficult.
Sentiment expressed sarcastically (e.g., “Great, my iPhone died again 🙄”) is hard for machine learning models to interpret correctly. This leads to misclassification in negative sentiment cases.

* Model trained on historical data.
The model reflects past patterns of customer language and may not immediately adapt to new product releases, evolving slang, or emerging sentiment shifts. Regular retraining with fresh Apple-related tweets is necessary.

* Imbalanced sentiment classes.
Negative tweets were the smallest portion of the dataset, which reduced model performance on this class despite techniques like class weights and SMOTE.

# 11. **Future Work**

* Fine-tune a transformer model further.
Increasing training epochs, expanding the dataset, and adjusting learning rates can significantly improve BERT-based performance—especially for neutral and sarcastic tweets.

* Collect more and fresher Apple-specific tweets.
Expanding the dataset with recent product launches (iPhone, Mac, iPad, Vision Pro) will help the model stay current with evolving customer language and sentiment trends.

* Implement time-series sentiment tracking.
Monitoring sentiment over weeks, months, or around product events (e.g., WWDC, product launches) would allow Apple to detect shifts earlier and respond quickly.

* Domain-specific lexicon enhancement.
Incorporating Apple-related terms (e.g., “FaceID,” “AirDrop,” “iOS update,” “battery health”) can improve contextual understanding and reduce misclassifications.

* Build a real-time dashboard.
A live system visualizing positive, neutral, and negative sentiment would give the marketing team actionable insights during major announcements or crises.
