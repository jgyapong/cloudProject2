# Utilize the designated Ubuntu base image
FROM ubuntu:20.04

#Configure the environment variable to disable user prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Install Java, which is necessary for Spark, along with any other requirements.
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget curl python3 python3-pip && \
    apt-get clean

# Install Spark
ENV SPARK_VERSION=3.1.2
ENV HADOOP_VERSION=3.2
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Configure the environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt


RUN mkdir -p /opt/spark/rf_model
# Transfer essential files to the Docker image
COPY rf_model /opt/spark/rf_model
COPY ValidationDataset_cleaned.csv /opt/spark/
COPY final_test.py /opt/spark/
COPY final_test_shell.sh /opt/spark/

# Specify the current directory
WORKDIR /opt/spark

# Specify the default command to be executed.
CMD ["bash", "final_test_shell.sh"]
