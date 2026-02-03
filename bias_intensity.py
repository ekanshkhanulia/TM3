# Component B roadmap (what we will do)
# 
# Load the processed dataset
# Use the CSV you already created (sentences_processed.csv).
#
#  Pick the label for Component B (bias intensity)
# Set y = mean_label (the 1–4 average score per sentence).
# 
# Filter the data for reliability
# Keep only sentences where:
# mean_label exists, and
# (recommended) agreement >= threshold (we’ll use 0.75 first, and you can also show 0.5 as comparison)
# 
# Prepare two different feature sets (two X’s)
# X_tone: tone features only (sentiment, subjectivity, strong-word counts, etc.)
# X_tfidf: TF-IDF vectors from the sentence text
# 
# Train the same regression model on both feature sets Use Ridge Regression for both:
# Ridge on X_tone → predict mean_label
# Ridge on X_tfidf → predict mean_label
# 
# Evaluate both models on the same test split Compare using:
# MAE (main)
# RMSE (optional)
# 
# Compare results side-by-side Make a small table:
# Tone-only MAE vs TF-IDF MAE
# Compute the gap (tone MAE − TF-IDF MAE)
# 
# Show a few disagreement examples Print a few sentences where:
# TF-IDF predicts much higher than tone-only (framing/content cases)
# tone-only predicts much higher than TF-IDF (pure tone cases)



import re 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge # a linear model with penalty
from sklearn.metrics import mean_absolute_error,mean_squared_error 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler


#load
df=pd.read_csv("processed/sentences_processed.csv")

print("Rows:" ,len(df))
print("col:" , list(df.columns)) 

#componenet
df=df[df["mean_label"].notna()].copy()
y=df["mean_label"].astype(float)

print("rows after keeping mean_label:" , len(df))


#filtter label
threshold=0.5 #142 rows outof 888 when we keep threshold to 0.75 so we droped our threshold to 0.5
df["reliable"]=df["agreement"] >= threshold

df=df[df["reliable"]==True].copy()
y=df["mean_label"].astype(float)

print("rows afeter reliability filter :",len(df))
print("y_samples :",len(y))


#input and target
X_text=df["text"]
y=df["mean_label"].astype(float)
print("X_text sameples:" ,len(X_text))
print("y_samples :",len(y))

#train and test split
X_train,X_test,y_train,y_test= train_test_split(X_text,y,test_size=0.2,random_state=42)

#tfidf  +ridge (model)
vectorizer=TfidfVectorizer()

X_train_vec=vectorizer.fit_transform(X_train) #learns and transform
X_test_vec=vectorizer.transform(X_test)

model_tfidf=Ridge()
model_tfidf.fit(X_train_vec,y_train)


y_pred=model_tfidf.predict(X_test_vec)

#evaluate
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred) ** 0.5

print("tfidfredge")
print("MAE:" ,mae)
print("Rmse:" ,rmse)







#tone only feature ridge

print("total !",
df["text"].str.count("!").sum()) # total 9 so no need to include
print("total ?",
df["text"].str.count(r"\?").sum()) # total 11 so no need to include


analyzer=SentimentIntensityAnalyzer()

def vader_features(text):
    scores= analyzer.polarity_scores(str(text))
    return [scores["neg"],scores["neu"],scores["pos"],scores["compound"]]

X_train_tone=[]
for t in X_train:
    X_train_tone.append(vader_features(t))

X_test_tone=[]
for t in X_test:
    X_test_tone.append(vader_features(t))

X_train_tone=np.array(X_train_tone,dtype=float)
X_test_tone=np.array(X_test_tone,dtype=float)

scaler=StandardScaler()
X_train_tone=scaler.fit_transform(X_train_tone) #
X_test_tone=scaler.transform(X_test_tone)


tone_model=Ridge()
tone_model.fit(X_train_tone,y_train)

#evaluate
y_pred_tone=tone_model.predict(X_test_tone)

mae_tone=mean_absolute_error(y_test,y_pred_tone)
rmse_tone=mean_squared_error(y_test,y_pred_tone) **0.5

print("Tone model (VAder +ridge)")
print("MAE",mae_tone)
print("rmse:",rmse_tone)


#intent like proxy (threshold)


X_all=df["text"]

X_all_vec=vectorizer.transform(X_all)
tfidf_pred_all=model_tfidf.predict(X_all_vec)



X_all_tone=[]
for t in X_all:
    X_all_tone.append(vader_features(t))

X_all_tone=np.array(X_all_tone,dtype=float)
X_all_tone=scaler.transform(X_all_tone)
tone_pred_all=tone_model.predict(X_all_tone)



df["pred_tfidf"] = tfidf_pred_all
df["pred_tone"] = tone_pred_all
df["persuasion_score"] = df["pred_tfidf"] - df["pred_tone"]

# Thresholds 
HUMAN_TH = 2.5
TONE_LOW = 2.0
TFIDF_HIGH = 2.5

df["intent_like_proxy"] = (
    (df["mean_label"] >= HUMAN_TH) &
    (df["pred_tone"] < TONE_LOW) &
    (df["pred_tfidf"] > TFIDF_HIGH)
)

print("\nComponent C results")
print("Total reliable sentences:", len(df))
print("Intent-like proxy sentences:", int(df["intent_like_proxy"].sum()))

# If source_bias exists
if "source_bias" in df.columns:
    print("\nIntent-like count by source_bias:")
    print(df[df["intent_like_proxy"]]["source_bias"].value_counts(dropna=False))

    print("\nIntent-like rate by source_bias:")
    print(df.groupby("source_bias")["intent_like_proxy"].mean())





#  percentile gap proxy





X_all_vec = vectorizer.transform(df["text"])
df["pred_tfidf"] = model_tfidf.predict(X_all_vec)


X_all_tone = []
for t in df["text"]:
    X_all_tone.append(vader_features(t))

X_all_tone = np.array(X_all_tone, dtype=float)
X_all_tone = scaler.transform(X_all_tone)   
df["pred_tone"] = tone_model.predict(X_all_tone)

# 2) Human based
biased_only = df[df["mean_label"] >= 2.5].copy()


print("Human-biased rows (mean_label>=2.5):", len(biased_only))

#  Compute gap and pick top 20%
biased_only["gap"] = biased_only["pred_tfidf"] - biased_only["pred_tone"]

top_gap_threshold = biased_only["gap"].quantile(0.80)
print("Top-gap threshold (90th percentile):", top_gap_threshold)

df["intent_like_proxy"] = False
df.loc[biased_only.index, "intent_like_proxy"] = biased_only["gap"] >= top_gap_threshold

print("Intent-like proxy count:", int(df["intent_like_proxy"].sum()))

# show distribution by outlet stance
print("\nIntent-like count by source_bias:")
print(df[df["intent_like_proxy"]]["source_bias"].value_counts(dropna=False))

print("\nIntent-like rate by source_bias:")
print(df.groupby("source_bias")["intent_like_proxy"].mean())

# print top 10 examples
top_examples = df[df["intent_like_proxy"]].copy()
top_examples["gap"] = top_examples["pred_tfidf"] - top_examples["pred_tone"]
top_examples = top_examples.sort_values("gap", ascending=False).head(10)

for _, r in top_examples.iterrows():
    print("--------------------")
    print("Sentence:", r["text"])
    print("mean_label:", r["mean_label"])
    print("TF-IDF:", round(float(r["pred_tfidf"]), 3), "Tone:", round(float(r["pred_tone"]), 3))
    print("gap:", round(float(r["gap"]), 3))
    print("source_bias:", r.get("source_bias", None))







#examples

import random
import json


test_df = df.loc[X_test.index, ["text", "labels_list", "mean_label"]].copy()


test_df["labels_list"] = test_df["labels_list"].apply(json.loads)


test_df["pred_tfidf"] = y_pred
test_df["pred_tone"] = y_pred_tone

# pick 5 random rows 
rows = test_df.sample(n=5, random_state=42)

for _, r in rows.iterrows():
    print("-----")
    print("Sentence:", r["text"])
    print("Human labels:", r["labels_list"])
    print("Mean label:", round(float(r["mean_label"]), 3))
    print("TF-IDF pred:", round(float(r["pred_tfidf"]), 3))
    print("Tone pred:", round(float(r["pred_tone"]), 3))





#C

