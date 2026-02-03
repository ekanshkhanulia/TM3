# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # sentiment (VADER)
# try:
#     from nltk.sentiment import SentimentIntensityAnalyzer
#     import nltk
#     try:
#         _ = SentimentIntensityAnalyzer()
#     except Exception:
#         nltk.download("vader_lexicon", quiet=True)
#     sia = SentimentIntensityAnalyzer()
# except Exception as e:
#     raise RuntimeError(
#         "VADER sentiment not available. Install nltk and download vader_lexicon."
#     ) from e


# # -----------------------------
# # config
# # -----------------------------
# CSV_PATH = "Sora_LREC2020_biasedsentences.csv"

# LABEL_MAP = {
#     "neutral": 0,
#     "neutral and not biased": 0,
#     "not biased": 0,
#     "slightly": 1,
#     "slightly biased": 1,
#     "slightly biased but acceptable": 1,
#     "biased": 2,
#     "very biased": 3,
#     "very biased–": 3,
#     "very biased-": 3,
#     "very biased.": 3,
# }

# LABEL_NAMES = {0: "neutral", 1: "slightly", 2: "biased", 3: "very biased"}


# def norm_label(x):
#     if x is None or (isinstance(x, float) and np.isnan(x)):
#         return np.nan
#     s = str(x).strip().lower()
#     s = s.replace("_", " ").replace("-", " ").replace("–", " ").strip()
#     s = " ".join(s.split())
#     for k, v in LABEL_MAP.items():
#         if s == k:
#             return v
#     if s in {"0", "1", "2", "3"}:
#         return int(s)
#     return np.nan


# def get_label_cols(df):
#     cols = []
#     if "t" in df.columns:
#         cols.append("t")
#     for i in range(0, 20):
#         c = str(i)
#         if c in df.columns:
#             cols.append(c)
#     return cols


# def get_sentence_cols(df):
#     cols = []
#     for i in range(0, 20):
#         c = f"s{i}"
#         if c in df.columns:
#             cols.append(c)
#     return cols


# # -----------------------------
# # load
# # -----------------------------
# df = pd.read_csv(CSV_PATH)

# label_cols = get_label_cols(df)
# sent_cols = get_sentence_cols(df)

# if "event" not in df.columns:
#     raise ValueError("Missing required column: event")
# if "reftext" not in df.columns:
#     raise ValueError("Missing required column: reftext")
# if "article_bias" not in df.columns:
#     raise ValueError("Missing required column: article_bias")
# if "preknow" not in df.columns:
#     raise ValueError("Missing required column: preknow")

# # numeric labels per cell
# for c in label_cols:
#     df[c] = df[c].apply(norm_label)

# # -----------------------------
# # Table 1: Event-wise label distribution (sentence-level)
# # -----------------------------
# long_labels = []
# for c in label_cols:
#     tmp = df[["event", c]].copy()
#     tmp = tmp.rename(columns={c: "bias"})
#     tmp["pos"] = c
#     long_labels.append(tmp)
# lab = pd.concat(long_labels, ignore_index=True)
# lab = lab.dropna(subset=["bias"])
# lab["bias"] = lab["bias"].astype(int)

# tab1_counts = pd.crosstab(lab["event"], lab["bias"])
# tab1_counts = tab1_counts.reindex(columns=[0, 1, 2, 3], fill_value=0)
# tab1_counts.columns = [LABEL_NAMES[i] for i in tab1_counts.columns]

# tab1_perc = tab1_counts.div(tab1_counts.sum(axis=1), axis=0).fillna(0) * 100
# tab1 = tab1_counts.astype(int).astype(str) + " (" + tab1_perc.round(1).astype(str) + "%)"

# print("\nTABLE 1: Event-wise sentence-label distribution (count + %)")
# print(tab1)

# # -----------------------------
# # Table 2: Position–Intensity summary
# # -----------------------------
# pos_mean = lab.groupby("pos")["bias"].mean().reindex(["t"] + [str(i) for i in range(20) if str(i) in label_cols])
# pos_std = lab.groupby("pos")["bias"].std().reindex(pos_mean.index)

# tab2 = pd.DataFrame({
#     "mean_bias_intensity(0-3)": pos_mean.round(3),
#     "std": pos_std.round(3),
#     "n_labels": lab.groupby("pos")["bias"].size().reindex(pos_mean.index).astype(int)
# })

# print("\nTABLE 2: Position–Intensity summary")
# print(tab2)

# # -----------------------------
# # Table 3: Annotator subjectivity (preknow)
# # simple: per-row avg bias score then group by preknow
# # -----------------------------
# df["row_bias_mean"] = df[label_cols].mean(axis=1, skipna=True)

# tab3 = df.groupby(df["preknow"].astype(str).str.lower().str.strip())["row_bias_mean"].agg(
#     n="count",
#     mean="mean",
#     variance="var",
#     std="std"
# ).round(4)

# print("\nTABLE 3: Preknow vs bias score (row-level mean across positions)")
# print(tab3)

# # -----------------------------
# # Plot 1: Tone vs Intensity heatmap (sentiment vs bias)
# # -----------------------------
# rows = []
# for i, s_col in enumerate(sent_cols):
#     lab_col = str(i)
#     if lab_col not in df.columns:
#         continue
#     tmp = df[[s_col, lab_col]].copy()
#     tmp = tmp.rename(columns={s_col: "text", lab_col: "bias"})
#     tmp = tmp.dropna(subset=["text", "bias"])
#     tmp["bias"] = tmp["bias"].astype(int)
#     rows.append(tmp)

# tone_df = pd.concat(rows, ignore_index=True)
# tone_df["sentiment"] = tone_df["text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])

# x = tone_df["sentiment"].to_numpy()
# y = tone_df["bias"].to_numpy()

# plt.figure()
# bins_x = np.linspace(-1, 1, 31)
# bins_y = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
# plt.hist2d(x, y, bins=[bins_x, bins_y])
# plt.yticks([0, 1, 2, 3], ["neutral", "slightly", "biased", "very biased"])
# plt.xlabel("Sentiment score (VADER compound)")
# plt.ylabel("Bias intensity (0-3)")
# plt.title("Tone vs Bias Intensity (Heatmap)")
# plt.colorbar(label="Sentence count")
# plt.tight_layout()

# # -----------------------------
# # Plot 2: Distance-from-reference vs article_bias (boxplot)
# # TF-IDF cosine distance between each sentence and reftext; aggregate to article mean drift
# # -----------------------------
# need_cols = ["id_article", "article_bias", "reftext"] + sent_cols
# if "id_article" not in df.columns:
#     raise ValueError("Missing required column: id_article")

# base = df[need_cols].copy()
# base = base.dropna(subset=["id_article", "reftext", "article_bias"])
# base["id_article"] = base["id_article"].astype(str)

# # use one row per article (reftext/doc sentences are same per article in your dataset structure)
# one = base.groupby("id_article", as_index=False).first()

# def mean_drift_for_row(row):
#     ref = str(row["reftext"])
#     sents = []
#     for c in sent_cols:
#         v = row.get(c, "")
#         if v is None or (isinstance(v, float) and np.isnan(v)):
#             continue
#         v = str(v).strip()
#         if v:
#             sents.append(v)
#     if not sents:
#         return np.nan

#     texts = [ref] + sents
#     vec = TfidfVectorizer(stop_words="english", max_features=5000)
#     X = vec.fit_transform(texts)
#     ref_vec = X[0]
#     sent_vecs = X[1:]
#     sims = cosine_similarity(sent_vecs, ref_vec).reshape(-1)
#     dists = 1.0 - sims
#     return float(np.mean(dists))

# one["drift_mean"] = one.apply(mean_drift_for_row, axis=1)
# one = one.dropna(subset=["drift_mean"])

# # normalize article_bias ordering
# one["article_bias_norm"] = one["article_bias"].astype(str).str.lower().str.strip()
# order = ["neutral", "slightly biased but acceptable", "slightly biased", "biased", "very biased"]
# present = [o for o in order if o in set(one["article_bias_norm"])]

# data = [one.loc[one["article_bias_norm"] == o, "drift_mean"].to_numpy() for o in present]

# plt.figure()
# plt.boxplot(data, labels=present, showfliers=False)
# plt.ylabel("Mean distance from reference (1 - cosine similarity)")
# plt.xlabel("article_bias")
# plt.title("Distance from Reference vs Article Bias")
# plt.tight_layout()

# plt.show()


















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


CSV_PATH = "Sora_LREC2020_biasedsentences.csv"

# Option B: dataset has labels 1..4; convert to 0..3 by subtracting 1
# 1=neutral, 2=slightly, 3=biased, 4=very biased  ->  0..3
def to_0_3(x):
    if pd.isna(x):
        return np.nan
    return int(x) - 1

LABELS_0_3 = [0, 1, 2, 3]
LABEL_NAMES = {0: "neutral", 1: "slightly", 2: "biased", 3: "very biased"}
ARTICLE_ORDER = ["neutral", "slightly biased but acceptable", "biased", "very biased"]


df = pd.read_csv(CSV_PATH)

label_cols = ["t"] + [str(i) for i in range(20)]
sent_cols = [f"s{i}" for i in range(20)]

for c in label_cols:
    df[c] = df[c].apply(to_0_3)


# TABLE 1: Event-wise label distribution (sentences)

lab = df.melt(id_vars=["event"], value_vars=label_cols, var_name="pos", value_name="bias").dropna()
lab["bias"] = lab["bias"].astype(int)

t1_counts = pd.crosstab(lab["event"], lab["bias"]).reindex(columns=LABELS_0_3, fill_value=0)
t1_perc = (t1_counts.div(t1_counts.sum(axis=1), axis=0) * 100).round(1)

t1 = t1_counts.astype(str) + " (" + t1_perc.astype(str) + "%)"
t1.columns = [LABEL_NAMES[i] for i in t1.columns]

print("\nTABLE 1: Event-wise sentence-label distribution ")
print(t1)


# TABLE 2: Position–Intensity 

pos_order = ["t"] + [str(i) for i in range(20)]
t2 = lab.groupby("pos")["bias"].agg(["mean", "std", "count"]).reindex(pos_order)
t2 = t2.rename(columns={"mean": "mean_bias(0-3)", "std": "std", "count": "n"}).round(3)

print("\nTABLE 2: Position–Intensity summary")
print(t2)


# TABLE 3: Annotator subjectivity (preknow) using row mean

df["row_bias_mean"] = df[label_cols].mean(axis=1, skipna=True)
t3 = df.groupby(df["preknow"].astype(str).str.lower().str.strip())["row_bias_mean"].agg(
    n="count", mean="mean", variance="var", std="std"
).round(4)

print("\nTABLE 3: Preknow vs bias score ")
print(t3)

