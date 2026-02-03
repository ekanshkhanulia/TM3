import argparse
import re
import json
import html
from collections import Counter

import pandas as pd

# cleaning

TAG_RE = re.compile(r"<[^>]+>")

def clean_text(text: str, strip_html: bool = True) -> str:
    # remove HTML tags if any, unescape entities, normalize whitespace
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # convert html things to normal
    s = html.unescape(s)

    # remove tags like <p>
    if strip_html:
        s = TAG_RE.sub(" ", s)  # replace with space

    # normalize whitespace and linebreaks
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def majority_label(labels: list[int]) -> int | None:
    "return the most common label; if tie, return smallest label"
    if not labels:
        return None
    c = Counter(labels)
    max_count = max(c.values())
    winners = [lab for lab, cnt in c.items() if cnt == max_count]
    return min(winners)


def majority_label_strict(labels: list[int]) -> int | None:
    "return majority label only if there is a unique winner; if tie return None"
    if not labels:
        return None
    c = Counter(labels)
    max_count = max(c.values())
    winners = [lab for lab, cnt in c.items() if cnt == max_count]
    if len(winners) != 1:
        return None
    return winners[0]


def agreement_fraction(labels: list[int], maj: int | None) -> float | None:
    "fraction of annotators that chose the majority label"
    if not labels or maj is None:
        return None
    return labels.count(maj) / len(labels)


def to_binary_label(x: int) -> int | None:
    # 1,2 -> 0 (low / not biased)
    # 3,4 -> 1 (biased)
    if x in (1, 2):
        return 0
    if x in (3, 4):
        return 1
    return None


# load raw csv
def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df



# Extract items (title + s0..s19) into long format

def to_long_format(df: pd.DataFrame, strip_html: bool = True) -> pd.DataFrame:
    required_cols = ["id_article", "doctitle", "t"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    meta_cols = [c for c in ["id_event", "source", "source_bias", "topic", "date"] if c in df.columns]

    rows = []

    for _, r in df.iterrows():
        # Title item
        title_text = clean_text(r["doctitle"], strip_html=strip_html)
        title_label = r["t"]
        if title_text:
            rows.append({
                "id_article": r["id_article"],
                "position": "title",
                "text": title_text,
                "label": title_label,
                **{c: r[c] for c in meta_cols},
            })

        # Sentece s0..s19 with labels 0..19
        for i in range(20):
            s_col = f"s{i}"
            l_col = str(i)

            if s_col not in df.columns or l_col not in df.columns:
                continue

            sent_text = clean_text(r[s_col], strip_html=strip_html)
            if not sent_text:
                continue

            rows.append({
                "id_article": r["id_article"],
                "position": i,
                "text": sent_text,
                "label": r[l_col],
                **{c: r[c] for c in meta_cols},
            })

    long_df = pd.DataFrame(rows)

    
    long_df["label"] = pd.to_numeric(long_df["label"], errors="coerce")
    long_df = long_df.dropna(subset=["label"])
    long_df["label"] = long_df["label"].astype(int)

    return long_df



#  compute summaries

def aggregate_annotations(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["id_article", "position"]

    # Keep first text/meta per group, collect all labels into list
    meta_keep = [c for c in long_df.columns if c not in group_cols + ["label"]]

    agg = (
        long_df
        .groupby(group_cols, as_index=False)
        .agg(
            text=("text", "first"),
            labels_list=("label", list),
            **{c: (c, "first") for c in meta_keep if c != "text"},
        )
    )

    # Original (1to4) summaries
    agg["n_annotators"] = agg["labels_list"].apply(len)
    agg["majority_label"] = agg["labels_list"].apply(majority_label)
    agg["mean_label"] = agg["labels_list"].apply(lambda x: sum(x) / len(x) if len(x) else None)
    agg["agreement"] = agg.apply(
        lambda row: agreement_fraction(row["labels_list"], row["majority_label"]),
        axis=1
    )

    # conservative): 1/2 -> 0, 3/4 -> 1
    
    def make_bin_list(labs):
        out = []
        for v in labs:
            b = to_binary_label(v)
            if b is not None:
                out.append(b)
        return out

    agg["labels_bin_list"] = agg["labels_list"].apply(make_bin_list)

    agg["majority_bin"] = agg["labels_bin_list"].apply(majority_label_strict)
    agg["agreement_bin"] = agg.apply(
        lambda row: agreement_fraction(row["labels_bin_list"], row["majority_bin"]),
        axis=1
    )

    agg["use_for_training"] = agg["agreement_bin"].apply(
        lambda x: True if (x is not None and x >= 0.75) else False
    )

    agg["y_bias_bin"] = agg["majority_bin"]

    # 
    # inclusive: 1 -> 0, 2/3/4 -> 1
    
    def to_binary_label_inclusive(x: int) -> int | None:
        if x == 1:
            return 0
        if x in (2, 3, 4):
            return 1
        return None

    def make_bin_list_inclusive(labs):
        out = []
        for v in labs:
            b = to_binary_label_inclusive(v)
            if b is not None:
                out.append(b)
        return out

    agg["labels_bin_list_inclusive"] = agg["labels_list"].apply(make_bin_list_inclusive)

    agg["majority_bin_inclusive"] = agg["labels_bin_list_inclusive"].apply(majority_label_strict)
    agg["agreement_bin_inclusive"] = agg.apply(
        lambda row: agreement_fraction(row["labels_bin_list_inclusive"], row["majority_bin_inclusive"]),
        axis=1
    )

    agg["use_for_training_inclusive"] = agg["agreement_bin_inclusive"].apply(
        lambda x: True if (x is not None and x >= 0.75) else False
    )

    agg["y_bias_bin_inclusive"] = agg["majority_bin_inclusive"]

    return agg



# EDA 

def make_eda_summary(agg_df: pd.DataFrame) -> dict:
    maj_counts = agg_df["majority_label"].value_counts(dropna=False).to_dict()
    agree_counts = agg_df["agreement"].round(2).value_counts().sort_index().to_dict()

    maj_bin_counts = agg_df["majority_bin"].value_counts(dropna=False).to_dict()
    agree_bin_counts = agg_df["agreement_bin"].round(2).value_counts(dropna=False).sort_index().to_dict()

    use_for_train_counts = agg_df["use_for_training"].value_counts(dropna=False).to_dict()

    return {
        "n_unique_items": int(len(agg_df)),
        "majority_label_distribution": {str(k): int(v) for k, v in maj_counts.items()},
        "agreement_distribution_rounded_2dp": {str(k): int(v) for k, v in agree_counts.items()},
        "n_annotators_distribution": {str(k): int(v) for k, v in agg_df["n_annotators"].value_counts().to_dict().items()},

        "majority_bin_distribution": {str(k): int(v) for k, v in maj_bin_counts.items()},
        "agreement_bin_distribution_rounded_2dp": {str(k): int(v) for k, v in agree_bin_counts.items()},
        "use_for_training_counts": {str(k): int(v) for k, v in use_for_train_counts.items()},
    }


#main
def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="processed/sentences_processed.csv")
    parser.add_argument("--out_eda_json", type=str, default="results/eda_summary.json")
    parser.add_argument(
        "--strip_html",
        action="store_true",
        help="Remove <p>...</p> tags and unescape HTML entities"
    )
    args = parser.parse_args()

    
    raw = load_raw(args.in_csv)
    long_df = to_long_format(raw, strip_html=args.strip_html)
    agg_df = aggregate_annotations(long_df)


    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    eda_dir = os.path.dirname(args.out_eda_json)
    if eda_dir:
        os.makedirs(eda_dir, exist_ok=True)

    # Save processed dataset as json
    agg_df_out = agg_df.copy()
    agg_df_out["labels_list"] = agg_df_out["labels_list"].apply(json.dumps)
    agg_df_out["labels_bin_list"] = agg_df_out["labels_bin_list"].apply(json.dumps)
    agg_df_out["labels_bin_list_inclusive"] = agg_df_out["labels_bin_list_inclusive"].apply(json.dumps)
    agg_df_out.to_csv(args.out_csv, index=False)

    
    eda = make_eda_summary(agg_df)
    with open(args.out_eda_json, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)

    print("Saved:", args.out_csv)
    print("EDA saved", args.out_eda_json)

if __name__ == "__main__":
    main()