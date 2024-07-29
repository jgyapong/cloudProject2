
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Instantiate the Spark session
spark = SparkSession.builder.appName("ModelTesting").getOrCreate()

# Define paths
model_path = "/home/ec2-user/rf_model"
validation_data_path = "/home/ec2-user/ValidationDataset_cleaned.csv"
output_file = "/home/ec2-user/f1_score.txt"

# Verify the existence of the model path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

# Verify the existence of the validation data path
if not os.path.exists(validation_data_path):
    raise FileNotFoundError(f"Validation data path '{validation_data_path}' does not exist.")

try:
    # Attempt to load the model as a PipelineModel first
    try:
        model = PipelineModel.load(model_path)
        print(f"PipelineModel loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"Failed to load as PipelineModel: {e}")
        #If the loading process as a PipelineModel fails, try loading it as a RandomForestClassificationModel.
        try:
            model = RandomForestClassificationModel.load(model_path)
            print(f"RandomForestClassificationModel loaded successfully from '{model_path}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # Import the validation data
    valid_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=';')
    print("Validation data loaded successfully.")

    # Verify the presence of the features column; if it does not exist, generate it
    required_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                        "density", "pH", "sulphates", "alcohol"]

    if set(required_columns).issubset(valid_data.columns):
        # Generate feature vector
        assembler = VectorAssembler(inputCols=required_columns, outputCol="features")
        valid_data = assembler.transform(valid_data)
    else:
        raise RuntimeError("Validation data does not contain the required columns.")

    # Generate forecasts for the validation dataset
    valid_predictions = model.transform(valid_data)
    print("Predictions made on validation data.")

    # Assess the model's performance using the F1 Score metric
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(valid_predictions)

    # Save the F1 score to a file
    with open(output_file, 'w') as f:
        f.write(f"F1 Score on Validation Data: {f1_score}")

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Terminate the Spark session
    spark.stop()

