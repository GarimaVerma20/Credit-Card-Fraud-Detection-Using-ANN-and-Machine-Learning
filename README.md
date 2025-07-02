
<div align="center">




</div>

# <div align="center">Credit card fraud detection using Neural networks</div>
<div align="center"><img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/intro.gif?raw=true"></div>



## Overview:
Our objective is to create the classifier for credit card fraud detection. To do it, we'll compare classification models from different methods :

- Logistic regression
- Support Vector Machine
- Bagging (Random Forest)
- Boosting (XGBoost)
- Neural Network (tensorflow/keras)
## Dataset:
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. I decided to proceed to an SMOTE strategy to re-balance the class.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.<br>
## Implementation:

**Libraries:**  `NumPy` `pandas` `pylab` `matplotlib` `sklearn` `seaborn` `plotly` `tensorflow` `keras` `imblearn`
## Data Exploration:
Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.<br>

The dataset is highly imbalanced ! It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, I will proceed to SMOTE.

It is a data preprocessing technique used to handle imbalanced classification problems, where one class (minority class) has significantly fewer samples than the other (majority class).

Instead of just duplicating existing minority class samples (which can cause overfitting), SMOTE generates new synthetic samples of the minority class to balance the dataset.

For SMOTE, we can use the package imblearn with SMOTE function. <br>

```
from imblearn.over_sampling import SMOTE

# Define SMOTE with a desired sampling strategy
smote = SMOTE(sampling_strategy=0.5)  # Generates synthetic samples until the minority class is 50% of the majority
```

<img src= "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/eda2.PNG?raw=true">

## Machine Learning Model Evaluation and Prediction:
### Logistic Regression:
<img src ="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr.PNG?raw=true" width="32%"> <img src= "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr1.PNG?raw=true" width="32%"> <img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/lr2.PNG?raw=true" width="32%">
```
Accuracy : 0.94
F1 score : 0.92
AUC : 0.96
```

### Support Vector Machine:
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm2.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/svm1.PNG?raw=true" width="32%">
```
Accuracy : 0.94
F1 score : 0.92
AUC : 0.97
```


### Random Forest:

<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF1.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/RF2.PNG?raw=true" width="32%">
```
Accuracy : 0.95
F1 score : 0.93
AUC : 0.97
```

### XGBoost:
The sequential ensemble methods, also known as “boosting”, creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence. The first model is built on training data, the second model improves the first model, the third model improves the second, and so on.<br>
<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG1.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/XG2.PNG?raw=true" width="32%">
```
Accuracy : 0.95
F1 score : 0.93
AUC : 0.97
```
### Multi Layer Perceptron:
<div align="center">
<img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/perceptron.PNG?raw=true">
</div>
The layers of a neural network are made of nodes. A node combines input from the data with a set of coefficients and bias, that either amplify or dampen that input, thereby assigning significance to inputs with regard to the task the algorithm is trying to learn. These input-weight products are summed and then the sum is passed through a node’s so-called activation function, to determine whether and to what extent that signal should progress further through the network to affect the ultimate outcome, say, an act of classification. If the signals passes through, the neuron has been “activated.” <br>

<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept1.PNG?raw=true" width="32%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/percept2.PNG?raw=true" width="32%">

```
Accuracy : 0.95
F1 score : 0.94
AUC : 0.98
```

### Neural Networks:
```
model = Sequential()
model.add(Dense(32, input_shape=(29,), activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(16, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(8, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(4, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001) #optimizer
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1,mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(X_train.values, y_train.values, epochs = 6, batch_size=5, validation_split = 0.15, verbose = 0, callbacks = [earlystopper])
```
The hidden layers are composed of an activation function called ReLU. It'is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. The last node has a sigmoid function that turns values to 0 or 1 (for binary classification).<br>

<img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/NN1.PNG?raw=true" width="40%"> <img src = "https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/NN2.PNG?raw=true" width="40%">
<img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN.PNG?raw=true" width="33%"> <img src ="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN1.PNG?raw=true" width="33%"> <img src="https://github.com/Pradnya1208/Credit-Card-Fraud-Detection-Using-Neural-Networks/blob/main/output/ANN2.PNG?raw=true" width="33%">
```
Accuracy : 0.95
F1 score : 0.94
AUC : 0.98
```





### Lessons Learned
`Neural Networks`
`Undersampling`
`Callbacks in Keras`
`Classification Algorithms`
`Multilayer Perceptrons`
`XGBoost classifier`
`Bagging`
`Boosting`





