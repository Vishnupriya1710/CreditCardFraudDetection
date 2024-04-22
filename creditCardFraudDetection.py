# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:32.781274Z","iopub.execute_input":"2024-04-22T19:45:32.781688Z","iopub.status.idle":"2024-04-22T19:45:36.290715Z","shell.execute_reply.started":"2024-04-22T19:45:32.781655Z","shell.execute_reply":"2024-04-22T19:45:36.289029Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:36.294392Z","iopub.execute_input":"2024-04-22T19:45:36.295638Z","iopub.status.idle":"2024-04-22T19:45:42.037804Z","shell.execute_reply.started":"2024-04-22T19:45:36.295551Z","shell.execute_reply":"2024-04-22T19:45:42.036384Z"}}
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()

# %% [markdown]
# 

# %% [markdown]
# Exploratory Data Analysis

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:42.045139Z","iopub.execute_input":"2024-04-22T19:45:42.045616Z","iopub.status.idle":"2024-04-22T19:45:42.092891Z","shell.execute_reply.started":"2024-04-22T19:45:42.045544Z","shell.execute_reply":"2024-04-22T19:45:42.091632Z"}}
data.info()

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:42.094935Z","iopub.execute_input":"2024-04-22T19:45:42.095280Z","iopub.status.idle":"2024-04-22T19:45:42.757292Z","shell.execute_reply.started":"2024-04-22T19:45:42.095251Z","shell.execute_reply":"2024-04-22T19:45:42.755521Z"}}
pd.set_option("display.float", "{:.2f}".format)
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:42.759228Z","iopub.execute_input":"2024-04-22T19:45:42.760756Z","iopub.status.idle":"2024-04-22T19:45:42.784432Z","shell.execute_reply.started":"2024-04-22T19:45:42.760714Z","shell.execute_reply":"2024-04-22T19:45:42.782835Z"}}
# Checking for null values
data.isnull().sum().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:42.786606Z","iopub.execute_input":"2024-04-22T19:45:42.787039Z","iopub.status.idle":"2024-04-22T19:45:42.800220Z","shell.execute_reply.started":"2024-04-22T19:45:42.786998Z","shell.execute_reply":"2024-04-22T19:45:42.796151Z"}}
data.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:42.802907Z","iopub.execute_input":"2024-04-22T19:45:42.803410Z","iopub.status.idle":"2024-04-22T19:45:43.347260Z","shell.execute_reply.started":"2024-04-22T19:45:42.803358Z","shell.execute_reply":"2024-04-22T19:45:43.345877Z"}}
LABELS = ["Normal","Fraud"]
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:43.349049Z","iopub.execute_input":"2024-04-22T19:45:43.352030Z","iopub.status.idle":"2024-04-22T19:45:43.366975Z","shell.execute_reply.started":"2024-04-22T19:45:43.351976Z","shell.execute_reply":"2024-04-22T19:45:43.365415Z"}}
data.Class.value_counts()

# %% [markdown]
# Determining the number of normal and fraud transactions

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:43.370687Z","iopub.execute_input":"2024-04-22T19:45:43.371208Z","iopub.status.idle":"2024-04-22T19:45:43.433238Z","shell.execute_reply.started":"2024-04-22T19:45:43.371174Z","shell.execute_reply":"2024-04-22T19:45:43.431516Z"}}
fraud = data[data['Class']==1]
normal = data[data['Class']==0]

print(f"Shape of Fraudulant transactions: {fraud.shape}")
print(f"Shape of Non-Fraudulant transactions: {normal.shape}")

# %% [markdown]
# Compare the money involved in the transactions

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:43.435025Z","iopub.execute_input":"2024-04-22T19:45:43.435507Z","iopub.status.idle":"2024-04-22T19:45:43.472133Z","shell.execute_reply.started":"2024-04-22T19:45:43.435470Z","shell.execute_reply":"2024-04-22T19:45:43.470834Z"}}
pd.concat([fraud.Amount.describe(), normal.Amount.describe()], axis=1)

# %% [markdown]
# Do transactions have a certain time frames?

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:45:43.473834Z","iopub.execute_input":"2024-04-22T19:45:43.475321Z","iopub.status.idle":"2024-04-22T19:45:43.504229Z","shell.execute_reply.started":"2024-04-22T19:45:43.475260Z","shell.execute_reply":"2024-04-22T19:45:43.502413Z"}}
pd.concat([fraud.Time.describe(), normal.Time.describe()], axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:48:47.608358Z","iopub.execute_input":"2024-04-22T19:48:47.609161Z","iopub.status.idle":"2024-04-22T19:49:14.519496Z","shell.execute_reply.started":"2024-04-22T19:48:47.609112Z","shell.execute_reply":"2024-04-22T19:49:14.518199Z"}}
# plot the time feature
plt.figure(figsize=(14,10))

plt.subplot(2, 2, 1)
plt.title('Time Distribution (Seconds)')

sns.displot(data['Time'], color='blue');

#plot the amount feature
plt.subplot(2, 2, 2)
plt.title('Distribution of Amount')
sns.displot(data['Amount'],color='blue');

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:49:45.384908Z","iopub.execute_input":"2024-04-22T19:49:45.385307Z","iopub.status.idle":"2024-04-22T19:49:46.537244Z","shell.execute_reply.started":"2024-04-22T19:49:45.385276Z","shell.execute_reply":"2024-04-22T19:49:46.535850Z"}}
# data[data.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6)
plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
data[data.Class == 1].Time.hist(bins=35, color='blue', alpha=0.6, label="Fraudulant Transaction")
plt.legend()

plt.subplot(2, 2, 2)
data[data.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6, label="Non Fraudulant Transaction")
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:50:19.907359Z","iopub.execute_input":"2024-04-22T19:50:19.907793Z","iopub.status.idle":"2024-04-22T19:50:21.894616Z","shell.execute_reply.started":"2024-04-22T19:50:19.907761Z","shell.execute_reply":"2024-04-22T19:50:21.893734Z"}}
# heatmap to find any high correlations

plt.figure(figsize=(10,10))
sns.heatmap(data=data.corr(), cmap="seismic")
plt.show();

# %% [markdown]
# Data Preprocessing

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T19:53:47.548465Z","iopub.execute_input":"2024-04-22T19:53:47.548996Z","iopub.status.idle":"2024-04-22T19:53:47.555859Z","shell.execute_reply.started":"2024-04-22T19:53:47.548961Z","shell.execute_reply":"2024-04-22T19:53:47.554505Z"}}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:02:50.978861Z","iopub.execute_input":"2024-04-22T20:02:50.979366Z","iopub.status.idle":"2024-04-22T20:02:51.406177Z","shell.execute_reply.started":"2024-04-22T20:02:50.979332Z","shell.execute_reply":"2024-04-22T20:02:51.404789Z"}}
scalar = StandardScaler()
X = data.drop('Class', axis=1)
y = data.Class

X_train_v, X_test, y_train_v, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v,test_size=0.2, random_state=42)

X_train = scalar.fit_transform(X_train)
X_validate = scalar.fit_transform(X_validate)
X_test = scalar.fit_transform(X_test)

w_p = y_train.value_counts()[0] / len(y_train)
w_n = y_train.value_counts()[1] / len(y_train)

print(f"Fraudulant transaction weight: {w_n}")
print(f"Non-Fraudulant transaction weight: {w_p}")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:03:03.358425Z","iopub.execute_input":"2024-04-22T20:03:03.359239Z","iopub.status.idle":"2024-04-22T20:03:03.367041Z","shell.execute_reply.started":"2024-04-22T20:03:03.359196Z","shell.execute_reply":"2024-04-22T20:03:03.365934Z"}}
print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n{'_'*55}")
print(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}\n{'_'*50}")
print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:10:36.451621Z","iopub.execute_input":"2024-04-22T20:10:36.453106Z","iopub.status.idle":"2024-04-22T20:10:36.463456Z","shell.execute_reply.started":"2024-04-22T20:10:36.453058Z","shell.execute_reply":"2024-04-22T20:10:36.461953Z"}}
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def print_score(label, prediction, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n") 

# %% [markdown]
# Model Building

# %% [markdown]
# Artificial Neural Network

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:12:43.779606Z","iopub.execute_input":"2024-04-22T20:12:43.780886Z","iopub.status.idle":"2024-04-22T20:12:58.505370Z","shell.execute_reply.started":"2024-04-22T20:12:43.780845Z","shell.execute_reply":"2024-04-22T20:12:58.504406Z"}}
from tensorflow import keras

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:40:07.577274Z","iopub.execute_input":"2024-04-22T20:40:07.577743Z","iopub.status.idle":"2024-04-22T20:40:07.724843Z","shell.execute_reply.started":"2024-04-22T20:40:07.577707Z","shell.execute_reply":"2024-04-22T20:40:07.723668Z"}}
model = keras.Sequential([
    keras.layers.Dense(256,activation='relu',input_shape=(X_train.shape[-1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T20:45:33.182826Z","iopub.execute_input":"2024-04-22T20:45:33.183345Z","iopub.status.idle":"2024-04-22T21:08:11.569960Z","shell.execute_reply.started":"2024-04-22T20:45:33.183301Z","shell.execute_reply":"2024-04-22T21:08:11.568594Z"}}
METRICS = [
#     keras.metrics.Accuracy(name='accuracy'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')
]

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=METRICS)

callbacks = [keras.callbacks.ModelCheckpoint('fraud_model_at_epoch_{epoch}.keras')]
class_weight = {0:w_p, 1:w_n}

r = model.fit(
    X_train, y_train, 
    validation_data=(X_validate, y_validate),
    batch_size=2048, 
    epochs=300, 
#     class_weight=class_weight,
    callbacks=callbacks,
)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:11:01.656770Z","iopub.execute_input":"2024-04-22T21:11:01.657309Z","iopub.status.idle":"2024-04-22T21:11:09.429832Z","shell.execute_reply.started":"2024-04-22T21:11:01.657271Z","shell.execute_reply":"2024-04-22T21:11:09.428323Z"}}
score = model.evaluate(X_test, y_test)
print(score)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:12:43.852648Z","iopub.execute_input":"2024-04-22T21:12:43.853628Z","iopub.status.idle":"2024-04-22T21:12:45.752815Z","shell.execute_reply.started":"2024-04-22T21:12:43.853553Z","shell.execute_reply":"2024-04-22T21:12:45.751706Z"}}
plt.figure(figsize=(12, 16))

plt.subplot(4, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='val_Loss')
plt.title('Loss Function evolution during training')
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(r.history['fn'], label='fn')
plt.plot(r.history['val_fn'], label='val_fn')
plt.title('Accuracy evolution during training')
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(r.history['precision'], label='precision')
plt.plot(r.history['val_precision'], label='val_precision')
plt.title('Precision evolution during training')
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(r.history['recall'], label='recall')
plt.plot(r.history['val_recall'], label='val_recall')
plt.title('Recall evolution during training')
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:13:48.695592Z","iopub.execute_input":"2024-04-22T21:13:48.696112Z","iopub.status.idle":"2024-04-22T21:14:11.391140Z","shell.execute_reply.started":"2024-04-22T21:13:48.696078Z","shell.execute_reply":"2024-04-22T21:14:11.389809Z"}}
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print_score(y_train, y_train_pred.round(), train=True)
print_score(y_test, y_test_pred.round(), train=False)

scores_dict = {
    'ANNs': {
        'Train': f1_score(y_train, y_train_pred.round()),
        'Test': f1_score(y_test, y_test_pred.round()),
    },
}

# %% [markdown]
# XGBoost

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:22:08.889561Z","iopub.execute_input":"2024-04-22T21:22:08.890125Z","iopub.status.idle":"2024-04-22T21:22:12.112570Z","shell.execute_reply.started":"2024-04-22T21:22:08.890089Z","shell.execute_reply":"2024-04-22T21:22:12.111230Z"}}
from xgboost import XGBClassifier

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train, eval_metric='aucpr')

y_train_pred = xgb_clf.predict(X_train)
y_test_pred = xgb_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

scores_dict['XGBoost'] = {
        'Train': f1_score(y_train,y_train_pred),
        'Test': f1_score(y_test, y_test_pred),
}

# %% [markdown]
# Random Forest

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:28:22.092871Z","iopub.execute_input":"2024-04-22T21:28:22.093542Z","iopub.status.idle":"2024-04-22T21:31:39.859431Z","shell.execute_reply.started":"2024-04-22T21:28:22.093503Z","shell.execute_reply":"2024-04-22T21:31:39.858104Z"}}
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, oob_score=False)
rf_clf.fit(X_train, y_train)

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

scores_dict['Random Forest'] = {
        'Train': f1_score(y_train,y_train_pred),
        'Test': f1_score(y_test, y_test_pred),
}

# %% [markdown]
# LightBGM

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:38:29.771345Z","iopub.execute_input":"2024-04-22T21:38:29.771870Z","iopub.status.idle":"2024-04-22T21:38:36.872238Z","shell.execute_reply.started":"2024-04-22T21:38:29.771835Z","shell.execute_reply":"2024-04-22T21:38:36.870897Z"}}
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier()
lgbm_clf.fit(X_train, y_train)

y_train_pred = lgbm_clf.predict(X_train)
y_test_pred = lgbm_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

scores_dict['LigthGBM'] = {
        'Train': f1_score(y_train,y_train_pred),
        'Test': f1_score(y_test, y_test_pred),
}

# %% [code] {"execution":{"iopub.status.busy":"2024-04-22T21:39:58.913923Z","iopub.execute_input":"2024-04-22T21:39:58.914957Z","iopub.status.idle":"2024-04-22T21:39:59.411786Z","shell.execute_reply.started":"2024-04-22T21:39:58.914918Z","shell.execute_reply":"2024-04-22T21:39:59.410567Z"}}
scores_df = pd.DataFrame(scores_dict)

scores_df.plot(kind='barh', figsize=(15, 8))

# %% [code]
