# Databricks notebook source
# MAGIC %md
# MAGIC #Titanic Survivor Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Let's read our dataset from s3, and explore provided variables

# COMMAND ----------

from pyspark.sql.functions import split, size
td = spark.read.csv('/mnt/ts/titanic.csv', header=True)
titanic_data = td.select(
  td.pclass.cast('int'),
  td.survived.cast('int'),
  td.name,
  td.sex,
  td.age.cast('float'),
  td.sibsp.cast('int'),
  td.parch.cast('int'),
  td.ticket,
  td.fare.cast('float'),
  td.cabin,
  td.embarked,
  td.boat,
  td.body,
  td.home_dest
)
titanic_data.registerTempTable('titanic')
titanic_data.cache()
display(titanic_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Display the summary for provided data

# COMMAND ----------

display(titanic_data.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ##Handle missing values
# MAGIC 
# MAGIC Handling missing values is a crucial part in feature engineering when working with data. There are many approaches to this issue, and we should defenitely try to avoid removing complete rows or columns from the dataset, due to the dataset size. We will try to impute missing values based on data distribution and prediction.
# MAGIC In this notebook we will go through the taken steps in handling missing values in Titanic Dataset. The following features with missing values were imputed in the dataset:
# MAGIC * Age
# MAGIC * Embarked
# MAGIC * Fare  
# MAGIC The features _ticket, cabin, boat, body and home\_destination_ will be unfortanutely removed from the dataset, since they contain many missing data fields, and imputing these values can give us false information based on the dataset size. So we are making a tough decision to work without these features.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Impute AGE
# MAGIC First column that we will deal with is _age_. Titanic Dataset is a very well known Kaggle competition, and many people have provided very clever and interesting ideas to deal with the dataset. While dealing with missing values, we will use ideas shared by [Megan L. Risdal on Kaggle](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results__.html), since this approach is quite interesting.
# MAGIC Since the dataset is small in size, we will do the age imputation in R, using [Mice](https://cran.r-project.org/web/packages/mice/index.html) package. We will impute missing age values without some of the columns that give us less useful information. The code for the R script is available in a file _ageImputation.R_.<br>
# MAGIC After running the code, we can compare the distribution of the age column before and after the imputation, just to make sure that we are not making any bad decisions:
# MAGIC ![Alt text](http://www.swiftpic.org/image.uploads/26-02-2017/original-8579452ad555b4ea57b9a155b38170ac.png)
# MAGIC 
# MAGIC After making sure the distribution will be the same after imutation phase, we can impute the missing age values.<br>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute Embarked
# MAGIC Based on some general ideas on the columns of the Titanic Dataset, prior knowledge and provided information, we can assume that _fare_ and _passenger class_ information can help us retrieve the missing embarkment information. From the dataset, we can see that both passengers that are missing embarkment information paid  $80 for the travel and that they belong to the pclass=1.
# MAGIC We can plot the information from which place other travelers with the same pclass and fare values usually embarked. The code for plotting the chart is available in a script _embarkedImputation.R_.
# MAGIC ![Alt text](http://www.swiftpic.org/image.uploads/26-02-2017/original-501bfc4a379d8bef96604e36961ace11.png)
# MAGIC <br>As we can see from the chart, other passengers that paid $80 (median fare) and belong to pclass=1 usually embarked from 'C'. Based on this information we can impute our missing values with for embarkment with 'C'. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute Fare
# MAGIC The final column with the missing value that we are going to deal with is _fare_ information. The passenger that is missing this information belongs to the third class, and embarked in 'S'. We can plot the median and distirbution of the _fare_ feature for the passengers that are from the same class and departed from the same place. (The R code is available in a script _fareImputation.R_) <br>
# MAGIC ![Alt text](http://www.swiftpic.org/image.uploads/26-02-2017/original-be2bcfd754f9b598aa2b80c840090a41.png) <br>
# MAGIC It seems reasonable based on the plot to replace the missing _fare_ value with median for their class and embarkment, which is $8.05.

# COMMAND ----------

# MAGIC %md
# MAGIC After dealing with the missing data in R, we are reading a final dataset, which contains all the imputed values.

# COMMAND ----------

td = spark.read.csv('/mnt/ts/titanic_final.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC After reading the dataset, we will retrieve state information from _home\_dest_ column

# COMMAND ----------

titanic_data = td.select(
  td.pclass.cast('int'),
  td.survived.cast('int'),
  td.name,
  td.sex,
  td.age.cast('float'),
  td.sibsp.cast('int'),
  td.parch.cast('int'),
  td.ticket,
  td.fare.cast('float'),
  td.cabin,
  td.embarked,
  td.boat,
  td.body,
  td.home_dest,
  split(split(td.home_dest, ('/'))[size(split(td.home_dest, ('/'))) -1], ',')[
    size(split(split(td.home_dest, ('/'))[size(split(td.home_dest, ('/'))) -1], ',')) - 1
  ].alias('state')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's register a temporary table so we can use SQL for creating required reports.

# COMMAND ----------

titanic_data.registerTempTable('titanic')

# COMMAND ----------

# MAGIC %sql
# MAGIC select pclass, count(1) as number_of_survivors
# MAGIC from titanic
# MAGIC where survived=1
# MAGIC group by pclass

# COMMAND ----------

# MAGIC %sql
# MAGIC select sex, count(1) as number_of_survivors
# MAGIC from titanic
# MAGIC where survived=1
# MAGIC group by sex

# COMMAND ----------

# MAGIC %sql
# MAGIC select age, count(1) as number_of_survivors
# MAGIC from titanic
# MAGIC where survived=1
# MAGIC group by age
# MAGIC order by age asc

# COMMAND ----------

# MAGIC %sql
# MAGIC select pclass, age, sex, count(*) as number_of_survivors
# MAGIC from titanic
# MAGIC where survived=1
# MAGIC group by pclass, age, sex
# MAGIC order by number_of_survivors desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select state, count(*) as number_of_survivors
# MAGIC from titanic
# MAGIC where survived=1
# MAGIC group by state
# MAGIC order by number_of_survivors desc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC Let's create some more features for our predictive model. We can see that each passenger has it's own title in a name, so let's retrieve titles for all our passengers, and count the occurences of each one of them.

# COMMAND ----------

titanic_data = titanic_data.withColumn(
  'title',
  split(split(titanic_data.name, ',')[1], ' ')[1]
)
display(titanic_data)

# COMMAND ----------

display(titanic_data.groupBy('title').count())

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that some titles are more frequent than the other ones, so we will create a categorical feature with following possible values:
# MAGIC * Mr.
# MAGIC * Mrs.
# MAGIC * Miss.
# MAGIC * Master.
# MAGIC * Other (for less frequent titles)  

# COMMAND ----------

from pyspark.sql import functions as F
titanic = titanic_data.withColumn('ptitle', F.when(titanic_data.title=='Mr.', 'Mr.').when(titanic_data.title=='Mrs.', 'Mrs.').when(titanic_data.title=='Miss.', 'Miss.').when(titanic_data.title=='Master.', 'Master.').otherwise('Other'))
titanic = titanic.drop('title').withColumnRenamed('ptitle', 'title')

# COMMAND ----------

display(titanic)

# COMMAND ----------

# MAGIC %md
# MAGIC We will stop at this point with Feature Engineering. There are many great ideas for retrieving more features at the official [Kaggle page of the contest](https://www.kaggle.com/c/titanic), and I am defenitely recommending taking a peak there.

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictive Modeling
# MAGIC Let's start building our predictive models. <br><br><br>
# MAGIC First step is to encode our categorical features and make feature vectors suitable for use in Spark ML pipelines and models.

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# display(encoded)
for x in ["pclass", "embarked", "title", "sex"]:
  indexer = StringIndexer(inputCol=x, outputCol=x+"Index").fit(titanic)
  indexed = indexer.transform(titanic)
  encoder = OneHotEncoder(dropLast=False, inputCol=x+"Index", outputCol=x+"Feature")
  titanic = encoder.transform(indexed)
display(titanic)

# COMMAND ----------

# MAGIC %md
# MAGIC We won't use all the available features in our models, we will focus on the most descriptive ones:
# MAGIC * age
# MAGIC * sibsp
# MAGIC * fare
# MAGIC * pclass
# MAGIC * embarked
# MAGIC * title
# MAGIC * sex

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["age", "sibsp", "parch", "fare", "pclassFeature", "embarkedFeature", "titleFeature", "sexFeature"],
    outputCol="features")

transformed = assembler.transform(titanic)
display(transformed)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC First model that we will test is Random Forest. The full documentation to the implementation of PySpark ML used is available in the official [PySpark documentation](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier).<br>
# MAGIC We will split our dataset into training and test datasets. <br>
# MAGIC Next step will be making a Pipeline that indexes all the labels in the label column and than trains the Random Forest model. <br>
# MAGIC After building the model, we'll test it on our test dataset.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Let's index our labels, and fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="survived", outputCol="label").fit(transformed)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = transformed.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, rf])

# Run labelIndexer and train the model.
model = pipeline.fit(trainingData)

# Let's Make predictions on our test data.
predictions = model.transform(testData)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC For model evaluation we will use PySpark's BinaryClassificationEvaluator, and Area Under ROC Curve measure. The Area Under the ROC Curve (AUC) is a measure of how well a parameter can distinguish between two groups.

# COMMAND ----------

# Let's evaluate our model using BinaryClassificationEvaluator. We will use Area Under ROC Curve measure to evaluate our model.
evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that our model behaves well with event the small number of trees (n=10)<br><br>
# MAGIC Let's visualize our results:

# COMMAND ----------

results = predictions.select(['probability', 'label'])
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

figure = plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
display(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC Based on this plot, we can say that our results are not really bad, since our ROC curve (blue line) is far from the diagonal dashed line. <br><br>
# MAGIC Let's see which features are the most important for our Random Forest model (we will use this information later on, in Feature Selection phase). <br>RandomForest.featureImportances computes, given a tree ensemble model, the importance of each feature.
# MAGIC 
# MAGIC This generalizes the idea of "Gini" importance to other losses, following the explanation of Gini importance from "Random Forests" documentation by Leo Breiman and Adele Cutler, and following the implementation from scikit-learn. The full explanation of the approach is documented in official [Spark documentation](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassificationModel.featureImportances).

# COMMAND ----------

model.stages[1].featureImportances

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Trees
# MAGIC Next model that we are going to use for our classification task is [Decision Tree](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-trees).
# MAGIC We will take the similar steps as in Random Forest implementation, and we will use the same datasets for training and testing.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Intialize a Decision Tree Classifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Make a Pipeline from Label Indexer and Decision Tree Model
pipeline = Pipeline(stages=[labelIndexer, dt])

# Run Label Indexer and train the model
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC Our Area under ROC curve for Decision Trees is not as good as with Random Forest alghorithm...

# COMMAND ----------

results1 = predictions.select(['probability', 'label'])
results_collect1 = results1.collect()
results_list1 = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect1]

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list1]
y_score = [i[0] for i in results_list1]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

figure = plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
display(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naive Bayes
# MAGIC Next model that we are going to test is [Naive Bayes](https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes), probabilistic classifiers based on applying Bayes’ theorem with strong (naive) independence assumptions between the features.
# MAGIC 
# MAGIC We are going to use multinomial classification, although we are trying to predict one of the two possible outcomes.

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

pipeline = Pipeline(stages=[labelIndexer, nb])

# Run Label Indexer and train the model
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC Naive Bayes is much more unprecise compared to Random Forest and Decison Trees

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

figure = plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
display(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC The final algorithm that we are going to try is [Logistic Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression). Logistic Regression is one of the most well-known algorithms that is used to predict one of the two possible outcomes.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(trainingData.select(trainingData.survived.alias('label'), trainingData.features))

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

predictions = lrModel.transform(testData.select(testData.survived, testData.features))
display(predictions)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression also shows less powerful results than Random Forest. 

# COMMAND ----------

display(predictions)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
results = predictions.select(['probability', 'survived'])
 
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)
 
metrics = metric(scoreAndLabels)
print("The ROC score is (@numTrees=10): ", metrics.areaUnderROC)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

figure = plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
display(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC After comparing the results of four tested models, we can see that we are getting quite satisfying results with Random Forest, even for the small number of trees. Random Forest is performing much better than the other tested algorithms. In that manner, we will try to select only the most relevant features, and try to lower the dimensionality of the problem. <br>
# MAGIC After evaluating the results of Random Forest, we have identified the importance of the used features:  
# MAGIC SparseVector(17, {0: 0.0589, 1: 0.0454, 2: 0.028, 3: 0.0834, 4: 0.1145, 5: 0.0469, 6: 0.0257, 7: 0.019, 8: 0.0159, 9: 0.003, 10: 0.0059, 11: 0.0009, 12: 0.0524, 13: 0.0211, 14: 0.0021, 15: 0.2103, 16: 0.2664}).<br>  
# MAGIC Here we are going to lower the dimensionality of the problem based on feature importance, and using Vector Slicer Feature Selecion. Let's test how our datasets behave with less features, and try to make more robust model while using smaller number of features.

# COMMAND ----------

from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row

slicer = VectorSlicer(inputCol="features", outputCol="selectedFeatures").setIndices([3, 4, 15, 16])

# We are using the same datasets as for the other algorithms
output = slicer.transform(transformed)
otestData = slicer.transform(testData)
otrainData = slicer.transform(trainingData)

# Let's make our model
rf = RandomForestClassifier(labelCol="label", featuresCol="selectedFeatures", numTrees=10)


# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(otrainData)

# Make predictions.
predictions = model.transform(otestData)

# Select example rows to display.
# display(predictions.select("prediction", "label", "features"))

# # Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md We have selected 4 of the most predictive features, and the results are better than results of other algorthms. <br>
# MAGIC Let's see will the performance grow significantly if we add one more feature.

# COMMAND ----------

from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row

# slicer = VectorSlicer(inputCol="features", outputCol="selectedFeatures").setIndices([3, 4, 15, 16])
slicer = VectorSlicer(inputCol="features", outputCol="selectedFeatures").setIndices([0, 3, 4, 15, 16])

output = slicer.transform(transformed)
otestData = slicer.transform(testData)
otrainData = slicer.transform(trainingData)

# display(output)
# train, test = output.randomSplit([0.7, 0.3])
# display(train)
rf = RandomForestClassifier(labelCol="label", featuresCol="selectedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(otrainData)

# Make predictions.
predictions = model.transform(otestData)

# Select example rows to display.
# display(predictions.select("prediction", "label", "features"))

# # Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)
print("Area under ROC = %g" % (areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC Our results from Random Forest with reduced number of dimensions to 5 shows almost the same performance as our model trained on all the 17 features (Area under ROC with 17 features was 0.869552, and we have managed to achieve Area under ROC of 0.86754 while using only 5 of the most predictive features).
# MAGIC ### Final Conclusion
# MAGIC In this notebook we have analyzed Titanic Dataset, and modeled the factors that are related to a passenger surviving the crash. <br>
# MAGIC In exploratory phase we have used R to impute the missing values, and explain our variables. <br><br>
# MAGIC For the purpose of Predictive Modeling, we have tried various classification approaches available in Spark 2.1 ML Library. Random Forest algorithm showed best performance, so we have decided to keep it as our predictive model for this purpose. We have decided to use Area under ROC curve as our metric for model performance, which shows us the power of the model to make a distinction betweeen two groups.<br>
# MAGIC After model exploration and model selection, we were dealing with model robustness. We have managed to represent predictive model with 5 dimensions instead of 17 that we have started with, and keep the model performance. <br>
# MAGIC Deployed model is scalable and robust, and it can easily work with significantly larger datasets (although the dataset for this problem can not grow significantly).<br><br>
# MAGIC Some possible improvements for the model might include creating more features from the original dataset, and trying to impute the other missing values or retrieve relevant information from these fields. 
