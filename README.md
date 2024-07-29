# cloudProject2
The objective of this assignment is to create parallel machine learning (ML) applications on the Amazon AWS cloud platform. You will acquire the following knowledge:

How to train an ML model in parallel on multiple EC2 instances using Apache Spark
.
How to develop and deploy an ML model in the cloud using Spark's MLlib.

How to facilitate deployment by utilizing Docker to generate a container for your machine learning model.

Training Data: TrainingDataset.csv - Utilized to train the model.

Validation Data: ValidationDataset.csv - Utilized to optimize and validate the efficacy of the model.

Test Data: TestDataset.csv - Utilized to evaluate the functionality and efficacy of your prediction application
.
Model Training and Evaluation: Executed in Python with Apache Spark. The script utilizes the training data to train a logistic regression model, validates its performance using the validation data, and subsequently stores the model.

The model prediction involves loading a trained model using a separate Python script, making predictions on new data, and evaluating the model's performance 

A Docker container is generated to encapsulate the prediction program. This facilitates seamless deployment across diverse contexts.

Provision 4 AWS EC2 instances for parallel training and 1 EC2 instance for prediction.

Apache Spark: Install and setup Spark on every EC2 instance.

Deploy Docker on the EC2 instance utilized for prediction.

Start the Spark cluster

Execute the Training Script

Compose a Dockerfile
