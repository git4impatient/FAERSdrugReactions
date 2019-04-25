from __future__ import print_function
#!pip install keras
#!pip install tensorflow
!echo $PYTHON_PATH
import os, sys
#import path
from pyspark.sql import *

# create spark sql session
myspark = SparkSession\
    .builder\
    .appName("FDA_drug_reactions") \
    .getOrCreate()



sc = myspark.sparkContext

import time
print ( time.time())

sc.setLogLevel("ERROR")
print ( myspark )
# make spark print text instead of octal
myspark.sql("SET spark.sql.parquet.binaryAsString=true")

# read in the data file from HDFS
fdadf = myspark.read.parquet ( "/user/hive/warehouse/fda.db/demo_normalized_p")
# also read from s3 mydf = myspark.read.parquet ( "s3a://impalas3a/sample_07_s3a_parquet")
drugdf = myspark.read.parquet ( "/user/hive/warehouse/fda.db/drugnormed_p")
cohortcountdf = myspark.read.parquet ( "/user/hive/warehouse/fda.db/cohortcounts")
cohortcountdf.show(5)

# print number of rows and type of object
print ( fdadf.count() )
print  ( fdadf )
fdadf.show(5)
drugdf.show(5)
# create a table name to use for queries
fdadf.createOrReplaceTempView("fdadata")
drugdf.createOrReplaceTempView("druglist")
cohortcountdf.createOrReplaceTempView("ccount")

# run a query
patientdf=myspark.sql('select * from fdadata where wt_n < 800')
patientdf.show(5)


# pairplot to see what we have...
import seaborn as sns
import pandas

## predict hospitalization or worse - cohort impact on areaUnderROC? 60-> .7299,  40 -> .7189,  20 -> .69179  10 ->.6834

#fda20k=myspark.sql('select fdadata.isr, druglist.isr , label, age_n, wt_n, gndr_n,druglisthash, medcount \
#  from fdadata, druglist where fdadata.isr= druglist.isr')
fda20k=myspark.sql('select label, age_n, wt_n, gndr_n,dl.druglisthash, medcount \
  from fdadata, druglist dl,ccount c where fdadata.isr= dl.isr \
  and c.druglisthash=dl.druglisthash and c.cohortcount>20')
fda20k.show(3)
fda20k.count()
# seaborn wants a pandas dataframe, not a spark dataframe
# so convert
pdsdf = fda20k.toPandas()

sns.set(style="ticks" , color_codes=True)
# this takes a long time to run:  
# you can see it if you uncomment it
g = sns.pairplot(pdsdf,  hue="medcount" )

#


# we can skip this step since we used Impala to make the 
# data numeric and normalize
# need to convert from text field to numeric
# this is a common requirement when using sparkML
#from pyspark.ml.feature import StringIndexer
# this will convert each unique string into a numeric
#indexer = StringIndexer(inputCol="txtlabel", outputCol="label")
#indexed = indexer.fit(mydf).transform(mydf)
#indexed.show(5)
# now we need to create  a  "label" and "features"
# input for using the sparkML library

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

## use  features to predict if there will be a fatality
assembler = VectorAssembler(
    inputCols=[ "age_n", "wt_n", "gndr_n", "druglisthash", "medcount"],
    outputCol="features")
output = assembler.transform(fda20k)
# note the column headers - label and features are keywords
print ( output.show(3) )
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)
# Print out the parameters, documentation, and any default values.
print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(output)

#### Major shortcut - no train and test data!!!
# Since model1 is a Model (i.e., a transformer produced by an Estimator),
# we can view the parameters it used during fit().
# This prints the parameter (name: value) pairs, where names are unique IDs for this
# LogisticRegression instance.
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

trainingSummary = model1.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

prediction = model1.transform(output)
prediction.show(3)
result = prediction.select("label", "probability", "prediction") \
    .collect()
#print(result)
i=0
for row in result:
   if ( row.label != row.prediction ):
    #print("label=%s, prob=%s, prediction=%s" \
    #      % (row.label, row.probability, row.prediction))
    i=i+1
    #if ( i > 10):
      #break
print ("total error count " )
print (i )      

trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))    

## can we do better with a deep learning keras network?
#https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
pdsdf.head()
pdsdf.describe(include='all')
sns.heatmap(pdsdf.corr(), annot=True)
# creating input features and target variables
X= pdsdf.iloc[:,1:8]
y= pdsdf.iloc[:,0]
X.head()
y.head()

# X_train=X.sample(frac=0.8,random_state=200)
# X_test=X.drop(X_train.index)

from sklearn.cross_validation import train_test_split

#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=5))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X,y, batch_size=10, epochs=100)
eval_model=classifier.evaluate(X, y)
eval_model

y_pred=classifier.predict(X)
y_pred =(y_pred>0.5)
# confusion matrix - barely correct when true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
print(cm)

# try a multiple output final stage?
