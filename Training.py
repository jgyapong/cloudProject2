from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Instantiate the Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load cleaned datasets
train_output_file = "/home/ec2-user/TrainingDataset_cleaned.csv"
valid_output_file = "/home/ec2-user/ValidationDataset_cleaned.csv"

train_data = spark.read.csv(train_output_file, header=True, inferSchema=True, sep=';')
valid_data = spark.read.csv(valid_output_file, header=True, inferSchema=True, sep=';')

# Eliminate rows containing null values
train_data = train_data.dropna()
valid_data = valid_data.dropna()

# Arrange the columns for features and labels.
feature_columns = [col for col in train_data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

# Process the data
train_data = assembler.transform(train_data).select('features', 'quality')
valid_data = assembler.transform(valid_data).select('features', 'quality')

# Normalize the characteristics
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaledFeatures", "quality")
valid_data = scaler_model.transform(valid_data).select("scaledFeatures", "quality")

print("Data prepared for training and validation.")

# Model utilizing the Random Forest Classifier algorithm
rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='quality')

# Perform model training
rf_model = rf.fit(train_data)

# Generate forecasts for the validation dataset
valid_predictions = rf_model.transform(valid_data)

# Assess the model
evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(valid_predictions)

print(f"Random Forest Classifier - F1 Score: {f1_score}")

# Assess additional measures
accuracy = evaluator.evaluate(valid_predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(valid_predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(valid_predictions, {evaluator.metricName: "weightedRecall"})

print(f"Random Forest Classifier - Accuracy: {accuracy}")
print(f"Random Forest Classifier - Precision: {precision}")
print(f"Random Forest Classifier - Recall: {recall}")

# Validating the application by using the validation dataset as the test dataset
test_data = valid_data

# Generate forecasts for the test dataset
test_predictions = rf_model.transform(test_data)

# Assess the model's performance using the F1 score
f1_score_test = evaluator.evaluate(test_predictions)

print(f"F1 Score on Test Data: {f1_score_test}")

# Preserve the model
rf_model.save("/home/ec2-user/rf_model")

# Initialize the model
loaded_model = RandomForestClassificationModel.load("/home/ec2-user/rf_model")

# Utilize the pre-trained model to provide predictions.
loaded_predictions = loaded_model.transform(test_data)

# Assess the model that has been loaded.
f1_score_loaded = evaluator.evaluate(loaded_predictions)

print(f"Loaded Model - F1 Score: {f1_score_loaded}")
print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
