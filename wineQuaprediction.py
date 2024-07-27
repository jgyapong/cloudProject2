
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import os

def set_spark_environment(spark_home, spark_bin):
	"""Set the Spark home and update the PATH."""
	os.environ['SPARK_HOME'] = spark_home
	os.environ['PATH'] += os.pathsep + spark_bin

def initialize_spark(app_name, master_url):
	"""Initialize and return a Spark session."""
	return SparkSession.builder \
		.appName(wineQuaprediction) \
		.master(master_url) \
		.getOrCreate()

def clean_column_names(df):
	"""Clean column names by removing spaces and quotes."""
	return df.toDF(*(c.replace(' ', '').replace('"', '') for c in df.columns))

def load_and_preprocess_data(spark, file_path, feature_cols):
	"""Load and preprocess data from a CSV file."""
	data = spark.read.csv(file_path, header=True, inferSchema=True)
	data = clean_column_names(data)
	assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
	return assembler.transform(data).select("features", "quality")

def train_logistic_regression(training_data):
	"""Train a Logistic Regression model."""
	lr = LogisticRegression(featuresCol='features', labelCol='quality')
	return lr.fit(training_data)

def evaluate_model(predictions):
	"""Evaluate the model using F1 score."""
	evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol='prediction', metricName='f1')
	return evaluator.evaluate(predictions)

def main():
	# Set Spark environment
	spark_home = '/home/ubuntu/spark-3.5.0-bin-hadoop3'
	spark_bin = os.path.join(spark_home, 'bin')
	set_spark_environment(spark_home, spark_bin)

	# Initialize Spark session
	master_url = "spark://<master-node-ip>:7077"  # Replace <master-node-ip> with your Spark master node IP
	spark = initialize_spark("Wine Quality Prediction", master_url)

	# Load and preprocess training data
	training_file_path = "s3://Dataset4/TrainingDataset.csv"
	training_data = spark.read.csv(training_file_path, header=True, inferSchema=True)
	training_data = clean_column_names(training_data)
	feature_cols = training_data.columns[:-1]  # Exclude label column
	training_data = load_and_preprocess_data(spark, training_file_path, feature_cols)

	# Train Logistic Regression model
	lr_model = train_logistic_regression(training_data)

	# Load and preprocess validation data
	validation_file_path = "s3://Dataset4/ValidationDataset.csv"
	validation_data = load_and_preprocess_data(spark, validation_file_path, feature_cols)

	# Make predictions
	predictions = lr_model.transform(validation_data)

	# Evaluate the model
	f1_score = evaluate_model(predictions)
	print(f"F1 Score: {f1_score}")

	# Save the model
	lr_model.save("/home/ubuntu/winePrediction/lr_model")

	# Stop the Spark session
	spark.stop()

if __name__ == "__main__":
	main()

