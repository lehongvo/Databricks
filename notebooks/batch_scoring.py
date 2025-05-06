# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Batch Scoring with Registered Model
# MAGIC 
# MAGIC This notebook demonstrates how to perform batch scoring (inference) using a model registered in the MLflow Model Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Parameters

# COMMAND ----------

# Define parameters
dbutils.widgets.text("model_name", "customer_churn_predictor", "Model Name")
dbutils.widgets.text("model_version", "latest", "Model Version")
dbutils.widgets.text("input_table", "prediction_input", "Input Table")
dbutils.widgets.text("output_table", "predictions", "Output Table")
dbutils.widgets.dropdown("output_mode", "overwrite", ["overwrite", "append", "merge"], "Output Mode")

# Read parameters
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
input_table = dbutils.widgets.get("input_table")
output_table = dbutils.widgets.get("output_table")
output_mode = dbutils.widgets.get("output_mode")

print(f"Model Name: {model_name}")
print(f"Model Version: {model_version}")
print(f"Input Table: {input_table}")
print(f"Output Table: {output_table}")
print(f"Output Mode: {output_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# Import required libraries
import mlflow
import mlflow.pyfunc
import mlflow.spark
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
import datetime
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Input Data

# COMMAND ----------

# Load the data to score
try:
    input_df = spark.table(input_table)
    print(f"Successfully loaded input table: {input_table}")
    print(f"Number of records to score: {input_df.count()}")
    
    # Display schema and sample records
    print("\nInput Schema:")
    input_df.printSchema()
    
    print("\nSample Records:")
    display(input_df.limit(5))
    
except Exception as e:
    print(f"Error loading input table: {str(e)}")
    
    # Create mock data if input table doesn't exist (for demonstration purposes)
    print("Creating mock data for demonstration...")
    
    # Define schema for mock data
    mock_schema = StructType([
        StructField("id", StringType(), False),
        StructField("category", StringType(), True),
        StructField("value", DoubleType(), True),
        StructField("value_normalized", DoubleType(), True),
        # Additional fields that match the expected input for the model
        StructField("feature1", DoubleType(), True),
        StructField("feature2", DoubleType(), True),
        StructField("feature3", StringType(), True)
    ])
    
    # Create mock data
    mock_data = [
        ("001", "A", 100.0, 1.0, 0.5, 0.7, "high"),
        ("002", "B", 85.0, 0.85, 0.3, 0.2, "medium"),
        ("003", "A", 120.0, 1.2, 0.8, 0.9, "high"),
        ("004", "C", 50.0, 0.5, 0.1, 0.3, "low"),
        ("005", "B", 95.0, 0.95, 0.4, 0.6, "medium")
    ]
    
    # Create DataFrame
    input_df = spark.createDataFrame(mock_data, mock_schema)
    
    # Write to a temporary table
    input_df.write.format("delta").mode("overwrite").saveAsTable(input_table)
    print(f"Created mock data table: {input_table}")
    
    # Display mock data
    print("\nMock Data Schema:")
    input_df.printSchema()
    
    print("\nMock Records:")
    display(input_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess the Input Data

# COMMAND ----------

# Function to preprocess data to match model expectations
def preprocess_data(df):
    """
    Preprocess the input data to match the format expected by the model.
    This will vary based on your model's requirements.
    """
    # Add preprocessing steps here
    # For example:
    # 1. Handle missing values
    processed_df = df.na.fill(0, ["value", "value_normalized"]) \
                     .na.fill("unknown", ["category"])
    
    # 2. Feature encoding (if needed)
    # For categorical variables that need to be encoded
    # This should match the preprocessing done during model training
    
    # 3. Feature scaling (if needed)
    # Scale numerical features if required
    
    # 4. Additional transformations
    # Add any other transformations required to match the model input format
    
    return processed_df

# Preprocess the input data
preprocessed_df = preprocess_data(input_df)
print("Data preprocessing completed.")
display(preprocessed_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Model from Registry

# COMMAND ----------

# Function to load model from MLflow model registry
def load_model(model_name, version="latest"):
    """
    Load a model from the MLflow model registry
    
    model_name: Name of the registered model
    version: Version of the model (can be "latest", "production", "staging", or a version number)
    """
    try:
        if version.lower() == "latest":
            model_uri = f"models:/{model_name}/latest"
        elif version.lower() in ["production", "staging"]:
            model_uri = f"models:/{model_name}/{version}"
        else:
            # Assume it's a version number
            model_uri = f"models:/{model_name}/{version}"
        
        # Log the model URI
        print(f"Loading model from: {model_uri}")
        
        # Try to load the model as a Spark UDF model first
        try:
            model = mlflow.spark.load_model(model_uri)
            model_type = "spark"
        except:
            # If that fails, try loading as a PyFunc model
            model = mlflow.pyfunc.load_model(model_uri)
            model_type = "pyfunc"
        
        print(f"Successfully loaded {model_type} model: {model_name} (version: {version})")
        return model, model_type
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        
        # For demonstration, create a mock model
        from pyspark.ml.classification import LogisticRegressionModel
        from pyspark.ml.feature import VectorAssembler
        
        print("Creating mock model for demonstration...")
        
        # Create a mock pipeline for demonstration
        assembler = VectorAssembler(inputCols=["value", "value_normalized"], outputCol="features")
        mock_df = assembler.transform(preprocessed_df)
        
        # Return the vector assembler as a "mock model" for demonstration
        return assembler, "mock"

# Load the model
model, model_type = load_model(model_name, model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the Data

# COMMAND ----------

# Function to score data using the loaded model
def score_data(df, model, model_type):
    """
    Score data using the loaded model
    
    df: DataFrame to score
    model: Loaded model
    model_type: Type of model ("spark", "pyfunc", or "mock")
    """
    try:
        # Add prediction timestamp
        df_with_timestamp = df.withColumn("scoring_timestamp", current_timestamp())
        
        # Score based on model type
        if model_type == "spark":
            # For Spark ML models
            predictions = model.transform(df_with_timestamp)
            
        elif model_type == "pyfunc":
            # For Python function models
            predictions = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{model_version}", result_type="double")
            predictions = df_with_timestamp.withColumn("prediction", predictions(*df.columns))
            
        elif model_type == "mock":
            # For the mock model (demonstration only)
            # Create a vector from selected features
            df_with_features = model.transform(df_with_timestamp)
            
            # Add a mock prediction column
            import pyspark.sql.functions as F
            import random
            
            # Create a UDF for mock predictions
            def mock_predict():
                return random.choice([0, 1])
            
            mock_predict_udf = udf(mock_predict, IntegerType())
            predictions = df_with_features.withColumn("prediction", mock_predict_udf())
            
            # Add probability column for completeness
            predictions = predictions.withColumn(
                "probability", 
                array(
                    F.when(col("prediction") == 0, F.rand()).otherwise(1 - F.rand()),
                    F.when(col("prediction") == 1, F.rand()).otherwise(1 - F.rand())
                )
            )
        
        print(f"Scoring completed successfully using {model_type} model.")
        return predictions
    
    except Exception as e:
        print(f"Error during scoring: {str(e)}")
        raise

# Score the data
scored_df = score_data(preprocessed_df, model, model_type)

# Display results
print("\nScoring Results:")
display(scored_df.select("id", "prediction", "probability", "scoring_timestamp").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Batch Predictions

# COMMAND ----------

# Basic evaluation of predictions (for demonstration)
# In a real scenario, you might have actual labels to compare against

# Get distribution of predictions
prediction_distribution = scored_df.groupBy("prediction").count().orderBy("prediction")
display(prediction_distribution)

# Create a visualization
import matplotlib.pyplot as plt
import pandas as pd

# Convert to pandas for easier plotting
pdf = prediction_distribution.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(pdf["prediction"].astype(str), pdf["count"])
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.grid(axis="y", alpha=0.75)

# Add count labels on top of bars
for i, v in enumerate(pdf["count"]):
    plt.text(i, v + 0.1, str(v), ha='center')

display(plt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Predictions to Output Table

# COMMAND ----------

# Function to save predictions to the output table
def save_predictions(predictions_df, table_name, mode="overwrite"):
    """
    Save the predictions to a Delta table
    
    predictions_df: DataFrame containing the predictions
    table_name: Name of the output table
    mode: Write mode (overwrite, append, merge)
    """
    try:
        if mode.lower() == "merge" and spark.catalog._jcatalog.tableExists(table_name):
            # If merge mode and table exists, perform a merge operation
            from delta.tables import DeltaTable
            
            # Get the Delta table
            delta_table = DeltaTable.forName(spark, table_name)
            
            # Define the merge condition (assuming 'id' is the primary key)
            merge_condition = "tgt.id = src.id"
            
            # Perform the merge operation
            delta_table.alias("tgt").merge(
                predictions_df.alias("src"),
                merge_condition
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
            
            print(f"Merged predictions into table: {table_name}")
        else:
            # Otherwise use standard write mode
            predictions_df.write.format("delta").mode(mode.lower()).saveAsTable(table_name)
            print(f"Saved predictions to table: {table_name} (mode: {mode})")
        
        # Get the count of records written
        count = spark.table(table_name).count()
        print(f"Total records in {table_name}: {count}")
        
        return True
    
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        return False

# Select columns to save
output_columns = ["id", "prediction", "probability", "scoring_timestamp"] + \
                [col for col in scored_df.columns if col not in ["prediction", "probability", "scoring_timestamp", "features"]]

# Save predictions to output table
save_success = save_predictions(
    scored_df.select(output_columns),
    output_table,
    output_mode
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Scoring Metrics

# COMMAND ----------

# Log scoring run information
scoring_metrics = {
    "model_name": model_name,
    "model_version": model_version,
    "scoring_time": datetime.datetime.now().isoformat(),
    "records_scored": scored_df.count(),
    "input_table": input_table,
    "output_table": output_table,
    "output_mode": output_mode,
    "prediction_distribution": prediction_distribution.toPandas().to_dict(orient="records")
}

# Convert to JSON for storage
scoring_metrics_json = json.dumps(scoring_metrics)

# Create a DataFrame for the metrics
metrics_df = spark.createDataFrame([(
    scoring_metrics["model_name"],
    scoring_metrics["model_version"],
    scoring_metrics["scoring_time"],
    scoring_metrics["records_scored"],
    scoring_metrics["input_table"],
    scoring_metrics["output_table"],
    scoring_metrics_json
)], ["model_name", "model_version", "scoring_time", "records_scored", "input_table", "output_table", "metrics_json"])

# Save the metrics
metrics_df.write.format("delta").mode("append").saveAsTable("model_scoring_metrics")
print("Scoring metrics saved to 'model_scoring_metrics' table.")

# Display metrics
display(metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Predictions (Optional)

# COMMAND ----------

# Function to export predictions to a file (CSV, JSON, etc.)
def export_predictions(predictions_df, export_path, format="csv"):
    """
    Export predictions to a file
    
    predictions_df: DataFrame containing the predictions
    export_path: Path to save the exported file
    format: Export format (csv, json, parquet, etc.)
    """
    try:
        predictions_df.write.format(format) \
            .option("header", "true") \
            .mode("overwrite") \
            .save(export_path)
        
        print(f"Exported predictions to {export_path} in {format} format")
        return True
    
    except Exception as e:
        print(f"Error exporting predictions: {str(e)}")
        return False

# Uncomment to export predictions
# export_path = "/dbfs/mnt/predictions/batch_export"
# export_predictions(scored_df.select(output_columns), export_path, "csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Up (Optional)

# COMMAND ----------

# Clean up temporary resources
# Uncomment if needed
# spark.sql(f"DROP TABLE IF EXISTS temp_scoring_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we have:
# MAGIC 1. Loaded a model from the MLflow Model Registry
# MAGIC 2. Preprocessed input data to match model requirements
# MAGIC 3. Performed batch scoring on the data
# MAGIC 4. Evaluated the prediction results
# MAGIC 5. Saved predictions to a Delta table
# MAGIC 6. Logged scoring metrics for monitoring
# MAGIC 
# MAGIC This batch scoring process can be scheduled as part of a production data pipeline. 