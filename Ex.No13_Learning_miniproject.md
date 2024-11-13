# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 24.10.24                                                                           
### REGISTER NUMBER : 212222040070
### AIM: 
The aim of this program is to build a machine learning model to classify messages as spam or ham, enhancing spam detection and improving message filtering accuracy.
###  Algorithm:
Step1:Data Preprocessing: Clean and tokenize text by removing punctuation, stop words, and converting to lowercase.

Step2:Feature Extraction: Create features like word count, character count, and apply TF-IDF for text representation.

Step3:Data Splitting: Split the dataset into training and testing sets to evaluate performance.

Step4:Model Training: Train a classification model (e.g., Naive Bayes) using the training data.

Step5:Model Evaluation: Assess the model’s accuracy and adjust as needed for improved spam detection.

### Program:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Warnings
import warnings
warnings.filterwarnings('ignore')

# Styles
plt.style.use('ggplot')
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['patch.force_edgecolor'] = True

import nltk
# nltk.download("all")
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

import spacy
nlp = spacy.load("en_core_web_sm") # Load the model with its full name

messages = pd.read_csv("/content/spam (1).csv", encoding = 'latin-1')

# Drop the extra columns and rename columns

messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["category", "text"]


display(messages.head(n = 10))


# Lets look at the dataset info to see if everything is alright

messages.info()

messages["category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

topMessages = messages.groupby("text")["category"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)
display(topMessages)

spam_messages = messages[messages["category"] == "spam"]["text"]
ham_messages = messages[messages["category"] == "ham"]["text"]

spam_words = []
ham_words = []

messages["messageLength"] = messages["text"].apply(len)
messages["messageLength"].describe()
```
### Output:
![image](https://github.com/user-attachments/assets/d954a5c0-7887-402e-8f58-a28ffe9be8ba)

![image](https://github.com/user-attachments/assets/be203ce9-c86d-418a-8d1e-5bc452db3849)

![image](https://github.com/user-attachments/assets/61b2d3a9-105e-4b8f-9d5a-da87283deb04)

![image](https://github.com/user-attachments/assets/22e2d4f5-66d0-4915-a4c2-e6c00f2e8219)

![image](https://github.com/user-attachments/assets/4d055ace-67da-43a9-a1b8-3198aaa8ce03)

![image](https://github.com/user-attachments/assets/c3958e87-c00e-4d8b-80bc-23737cfd9314)


### Result:
Thus the system was trained successfully and the prediction was carried out.
