# #2.1 Load data (processed/sentences_processed.csv)
# 2.2 Filter reliable data (use_for_training == True)

# 2.3 Define X and y
# X = text
# y = y_bias_bin

# 2.4 Vectorize text (TF-IDF)
# 2.5 Train baseline model (LogReg or Linear SVM)
# 2.6 Evaluate (Accuracy, Precision, Recall, F1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB




print("\n===== where binary labels are : (1,2->0 | 3,4->1) =====")
#load data
df=pd.read_csv("processed/sentences_processed.csv")
print ("Rows:" , len(df))
print ("columns:", list(df.columns))

#filter data

df=df[df["use_for_training"] == True].copy()

print("rows after filtering",len(df))
print(df["y_bias_bin"].value_counts(dropna=False)) #counts 0 and 1 in the sentences_processed.csv also counting missing values if exist 





#training (define x and y )
X=df["text"].tolist()  #input (sentences text)
y=df['y_bias_bin'].tolist()  # output to learn

print("X samples:", len(X))
print("y samples:", len(y))

#vectorize text
vectorizer=TfidfVectorizer()

X_vec = vectorizer.fit_transform(X) #term freq-> vocab-> weight

print("vectoirzed shape" , X_vec.shape)


#train baseline (LR , svm, NaiveB)
X_train,X_test, y_train,y_test= train_test_split(X_vec, y, test_size=0.2,random_state=42, stratify=y)  #spilt

logreg= LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)

svm =LinearSVC()
svm.fit(X_train,y_train)

nb=MultinomialNB()
nb.fit(X_train,y_train)



#evaluate

def evaluate(model,X_test,y_test,name):
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    p,r,f,_=precision_recall_fscore_support(y_test,y_pred,average="binary",zero_division=0)
    print(name)
    print("accuracy",acc)
    print("precesion:" ,p)
    print("recall:",r)
    print("f1:",f)
    print()
    

evaluate(logreg,X_test,y_test,"Logistic Regression")
evaluate(svm,X_test,y_test,"Linear Svm")
evaluate(nb,X_test,y_test,"Naive Bayes")





print("\n INCLUSIVE (1->0 | 2,3,4->1) ")

# load again (or reuse original df before filtering if you want)
df2 = pd.read_csv("processed/sentences_processed.csv")

# filter inclusive
df2 = df2[df2["use_for_training_inclusive"] == True].copy()

print("rows after filtering", len(df2))
print(df2["y_bias_bin_inclusive"].value_counts(dropna=False))

# define X and y
X2 = df2["text"].tolist()
y2 = df2["y_bias_bin_inclusive"].tolist()

print("X samples:", len(X2))
print("y samples:", len(y2))

# vectorize
vectorizer2 = TfidfVectorizer()
X2_vec = vectorizer2.fit_transform(X2)

print("vectorized shape", X2_vec.shape)

# split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_vec, y2, test_size=0.2, random_state=42, stratify=y2
)

# train models
logreg2 = LogisticRegression(max_iter=1000)
logreg2.fit(X2_train, y2_train)

svm2 = LinearSVC()
svm2.fit(X2_train, y2_train)

nb2 = MultinomialNB()
nb2.fit(X2_train, y2_train)

# evaluate (reuse your existing evaluate function)
evaluate(logreg2, X2_test, y2_test, "Logistic Regression (Inclusive)")
evaluate(svm2, X2_test, y2_test, "Linear SVM (Inclusive)")
evaluate(nb2, X2_test, y2_test, "Naive Bayes (Inclusive)")



print("\n CLASS WEIGHT BALANCED (Conservative Mapping) ")

# Logistic Regression with class_weight='balanced'
logreg_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_bal.fit(X_train, y_train)

# Linear SVM with class_weight='balanced'
svm_bal = LinearSVC(class_weight='balanced')
svm_bal.fit(X_train, y_train)

# Evaluate
evaluate(logreg_bal, X_test, y_test, "Logistic Regression (Balanced)")
evaluate(svm_bal, X_test, y_test, "Linear SVM (Balanced)")


print("\nCLASS WEIGHT BALANCED (Inclusive Mapping)")

# Logistic Regression with class_weight='balanced' on inclusive mapping
logreg2_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg2_bal.fit(X2_train, y2_train)

# Linear SVM with class_weight='balanced' on inclusive mapping
svm2_bal = LinearSVC(class_weight='balanced')
svm2_bal.fit(X2_train, y2_train)

# Evaluate
evaluate(logreg2_bal, X2_test, y2_test, "Logistic Regression (Inclusive + Balanced)")
evaluate(svm2_bal, X2_test, y2_test, "Linear SVM (Inclusive + Balanced)")