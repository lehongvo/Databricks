# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Machine Learning Pipeline
# MAGIC 
# MAGIC This notebook demonstrates how to build a complete machine learning pipeline in Databricks, from data preparation to model deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# Import necessary libraries
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Enable MLflow autologging
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading

# COMMAND ----------

# Load the processed data from Delta table
df = spark.table("processed_data")
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Define target variable and features
target_col = "target"  # Assuming 'target' is a binary classification column

# Identify categorical and numerical columns
categorical_cols = ["category"]  # Add all categorical columns here
numerical_cols = ["value", "value_normalized"]  # Add all numerical columns here

# Create a pipeline for preprocessing
stages = []

# Process categorical columns
for cat_col in categorical_cols:
    # Convert string columns to category indices
    string_indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index", handleInvalid="keep")
    # Convert indices to one-hot encodings
    encoder = OneHotEncoder(inputCols=[f"{cat_col}_index"], outputCols=[f"{cat_col}_encoded"])
    stages += [string_indexer, encoder]

# Combine all features into a single vector
feature_cols = [f"{col}_encoded" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
stages += [assembler]

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
stages += [scaler]

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline(stages=stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split

# COMMAND ----------

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training set size: {train_df.count()}")
print(f"Test set size: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training with MLflow Tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression

# COMMAND ----------

# Start an MLflow run for tracking
with mlflow.start_run(run_name="Logistic Regression Model"):
    
    # Fit the preprocessing pipeline on the training data
    preprocessing_model = preprocessing_pipeline.fit(train_df)
    train_processed = preprocessing_model.transform(train_df)
    test_processed = preprocessing_model.transform(test_df)
    
    # Create a logistic regression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol=target_col)
    
    # Train the model
    lr_model = lr.fit(train_processed)
    
    # Make predictions on test data
    predictions = lr_model.transform(test_processed)
    
    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
    auc_roc = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("areaUnderPR")
    auc_pr = evaluator.evaluate(predictions)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("regularization", lr.getRegParam())
    mlflow.log_param("elasticNetParam", lr.getElasticNetParam())
    
    mlflow.log_metric("AUC ROC", auc_roc)
    mlflow.log_metric("AUC PR", auc_pr)
    
    print(f"Logistic Regression AUC-ROC: {auc_roc:.4f}")
    print(f"Logistic Regression AUC-PR: {auc_pr:.4f}")
    
    # Save the model
    mlflow.spark.log_model(lr_model, "lr_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

# Start an MLflow run for tracking Random Forest
with mlflow.start_run(run_name="Random Forest Model"):
    
    # Create a random forest classifier
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=target_col, numTrees=100)
    
    # Train the model
    rf_model = rf.fit(train_processed)
    
    # Make predictions on test data
    predictions = rf_model.transform(test_processed)
    
    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
    auc_roc = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("areaUnderPR")
    auc_pr = evaluator.evaluate(predictions)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("numTrees", rf.getNumTrees())
    mlflow.log_param("maxDepth", rf.getMaxDepth())
    
    mlflow.log_metric("AUC ROC", auc_roc)
    mlflow.log_metric("AUC PR", auc_pr)
    
    print(f"Random Forest AUC-ROC: {auc_roc:.4f}")
    print(f"Random Forest AUC-PR: {auc_pr:.4f}")
    
    # Feature importance
    feature_importances = rf_model.featureImportances.toArray()
    
    # Log the model
    mlflow.spark.log_model(rf_model, "rf_model")
    
    # Create and log feature importance plot
    feature_names = feature_cols
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    
    # Save and log the figure
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

# Start an MLflow run for hyperparameter tuning
with mlflow.start_run(run_name="GBT Hyperparameter Tuning"):
    
    # Create a GBT classifier
    gbt = GBTClassifier(featuresCol="scaled_features", labelCol=target_col)
    
    # Create parameter grid for tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [3, 5, 7]) \
        .addGrid(gbt.maxIter, [10, 20]) \
        .addGrid(gbt.stepSize, [0.05, 0.1]) \
        .build()
    
    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    
    # Run cross-validation and select the best model
    cv_model = cv.fit(train_processed)
    
    # Get the best model
    best_model = cv_model.bestModel
    
    # Make predictions with the best model
    predictions = best_model.transform(test_processed)
    
    # Evaluate the best model
    auc_roc = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("areaUnderPR")
    auc_pr = evaluator.evaluate(predictions)
    
    # Log best parameters and metrics
    mlflow.log_param("model_type", "GBT")
    mlflow.log_param("best_maxDepth", best_model.getMaxDepth())
    mlflow.log_param("best_maxIter", best_model.getMaxIter())
    mlflow.log_param("best_stepSize", best_model.getStepSize())
    
    mlflow.log_metric("Best AUC ROC", auc_roc)
    mlflow.log_metric("Best AUC PR", auc_pr)
    
    print(f"Best GBT model AUC-ROC: {auc_roc:.4f}")
    print(f"Best GBT model AUC-PR: {auc_pr:.4f}")
    
    # Log the best model
    mlflow.spark.log_model(best_model, "gbt_best_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

# Query MLflow for model metrics
runs = mlflow.search_runs(filter_string="metric.AUC ROC > 0")
display(runs[["run_id", "params.model_type", "metrics.AUC ROC", "metrics.AUC PR"]])

# Create a plot to compare model performance
model_names = runs["params.model_type"].tolist()
auc_roc_values = runs["metrics.AUC ROC"].tolist()
auc_pr_values = runs["metrics.AUC PR"].tolist()

plt.figure(figsize=(12, 6))
x = range(len(model_names))
width = 0.35

plt.bar(x, auc_roc_values, width, label='AUC ROC')
plt.bar([i + width for i in x], auc_pr_values, width, label='AUC PR')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks([i + width/2 for i in x], model_names)
plt.legend()
plt.ylim(0, 1)

plt.savefig("model_comparison.png")
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Deployment

# COMMAND ----------

# Register the best model in the MLflow Model Registry
best_run_id = runs.loc[runs["metrics.AUC ROC"].idxmax(), "run_id"]
model_uri = f"runs:/{best_run_id}/gbt_best_model"

# Register the model
model_name = "customer_churn_predictor"
model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Registered model '{model_name}' as version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference

# COMMAND ----------

# Load the best model from the registry
model_version_uri = f"models:/{model_name}/latest"
loaded_model = mlflow.spark.load_model(model_version_uri)

# Generate predictions on new data (using test data as an example)
batch_predictions = loaded_model.transform(test_processed)
display(batch_predictions.select("id", target_col, "probability", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Monitoring Setup

# COMMAND ----------

# Save predictions for monitoring
batch_predictions.write.format("delta").mode("overwrite").saveAsTable("model_predictions")

# Example code to generate model monitoring metrics
monitoring_metrics = batch_predictions.selectExpr(
    "prediction", 
    target_col,
    f"CASE WHEN prediction = {target_col} THEN 1 ELSE 0 END as correct"
)

# Calculate accuracy
accuracy = monitoring_metrics.agg(avg("correct").alias("accuracy")).collect()[0]["accuracy"]
print(f"Model Accuracy: {accuracy:.4f}")

# Save monitoring metrics for dashboard
monitoring_metrics.write.format("delta").mode("overwrite").saveAsTable("model_monitoring_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we've demonstrated a complete machine learning workflow:
# MAGIC 1. Data preparation and feature engineering
# MAGIC 2. Training multiple models (Logistic Regression, Random Forest, GBT)
# MAGIC 3. Hyperparameter tuning with cross-validation
# MAGIC 4. Tracking experiments with MLflow
# MAGIC 5. Model comparison and selection
# MAGIC 6. Model deployment via MLflow Model Registry
# MAGIC 7. Batch inference
# MAGIC 8. Basic model monitoring 