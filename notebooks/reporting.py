# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Reporting Dashboard
# MAGIC 
# MAGIC This notebook generates reports and visualizations based on prediction results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Parameters

# COMMAND ----------

# Define parameters
dbutils.widgets.text("data_table", "predictions", "Data Table")
dbutils.widgets.text("report_path", "/reports/weekly_summary", "Report Path")
dbutils.widgets.dropdown("time_period", "week", ["day", "week", "month"], "Time Period")

# Read parameters
data_table = dbutils.widgets.get("data_table")
report_path = dbutils.widgets.get("report_path")
time_period = dbutils.widgets.get("time_period")

print(f"Data Table: {data_table}")
print(f"Report Path: {report_path}")
print(f"Time Period: {time_period}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# Import necessary libraries
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load prediction data
try:
    predictions_df = spark.table(data_table)
    print(f"Successfully loaded predictions table: {data_table}")
    print(f"Number of records: {predictions_df.count()}")
    
    # Sample the data
    display(predictions_df.limit(5))
    
except Exception as e:
    print(f"Error loading predictions table: {str(e)}")
    
    # Create mock data for demonstration
    print("Creating mock prediction data for demonstration...")
    
    # Generate dates
    current_date = datetime.now()
    
    if time_period == "day":
        date_range = [current_date - timedelta(hours=i) for i in range(24)]
    elif time_period == "week":
        date_range = [current_date - timedelta(days=i) for i in range(7)]
    else:  # month
        date_range = [current_date - timedelta(days=i) for i in range(30)]
    
    # Create mock data
    mock_data = []
    categories = ["A", "B", "C"]
    
    for i in range(100):
        category = categories[i % len(categories)]
        prediction = i % 2
        score_date = date_range[i % len(date_range)]
        
        mock_data.append((
            f"ID-{i:03d}",
            category,
            float(100 + i),
            float((100 + i) / 100),
            prediction,
            [0.8 if prediction == 1 else 0.2, 0.2 if prediction == 1 else 0.8],
            score_date
        ))
    
    # Create DataFrame
    predictions_df = spark.createDataFrame(mock_data, [
        "id", 
        "category", 
        "value", 
        "value_normalized", 
        "prediction", 
        "probability",
        "scoring_timestamp"
    ])
    
    # Save to a temporary table
    predictions_df.write.format("delta").mode("overwrite").saveAsTable(data_table)
    print(f"Created mock prediction data in table: {data_table}")
    
    # Display mock data
    display(predictions_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data For Reporting

# COMMAND ----------

# Add time dimensions for reporting
reporting_df = predictions_df.withColumn("prediction_date", to_date("scoring_timestamp")) \
    .withColumn("prediction_hour", hour("scoring_timestamp")) \
    .withColumn("prediction_day", dayofmonth("scoring_timestamp")) \
    .withColumn("prediction_month", month("scoring_timestamp")) \
    .withColumn("prediction_year", year("scoring_timestamp")) \
    .withColumn("day_of_week", date_format("scoring_timestamp", "E"))

# Calculate positive prediction probability
reporting_df = reporting_df.withColumn(
    "positive_probability", 
    when(col("prediction") == 1, element_at(col("probability"), 1)).otherwise(element_at(col("probability"), 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Summary Statistics

# COMMAND ----------

# Calculate summary statistics
summary_stats = reporting_df.groupBy("prediction").agg(
    count("*").alias("count"),
    avg("value").alias("avg_value"),
    min("value").alias("min_value"),
    max("value").alias("max_value"),
    avg("positive_probability").alias("avg_probability")
)

display(summary_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Analysis

# COMMAND ----------

# Group by date and prediction
time_series = reporting_df.groupBy("prediction_date", "prediction") \
    .agg(count("*").alias("count")) \
    .orderBy("prediction_date", "prediction")

# Convert to pandas for visualization
time_series_pd = time_series.toPandas()

# Create a time series plot
plt.figure(figsize=(12, 6))
for prediction, group in time_series_pd.groupby("prediction"):
    plt.plot(group["prediction_date"], group["count"], label=f"Prediction {prediction}")

plt.title("Predictions Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

display(plt)

# Save the plot
# plt.savefig("/tmp/time_series.png")
# dbutils.fs.cp("file:/tmp/time_series.png", f"{report_path}/time_series.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Category Analysis

# COMMAND ----------

# Group by category and prediction
category_analysis = reporting_df.groupBy("category", "prediction") \
    .agg(count("*").alias("count")) \
    .orderBy("category", "prediction")

# Display as a table
display(category_analysis)

# Convert to pandas for visualization
category_pd = category_analysis.toPandas()

# Create a pivot table
category_pivot = category_pd.pivot(index="category", columns="prediction", values="count").fillna(0)

# Plot a stacked bar chart
plt.figure(figsize=(10, 6))
category_pivot.plot(kind="bar", stacked=True)
plt.title("Predictions by Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.legend(title="Prediction")
plt.grid(axis="y", alpha=0.3)

display(plt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Value Distribution Analysis

# COMMAND ----------

# Convert to Pandas for easier visualization
value_dist_pd = reporting_df.select("value", "prediction").toPandas()

# Create box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x="prediction", y="value", data=value_dist_pd)
plt.title("Value Distribution by Prediction")
plt.grid(axis="y", alpha=0.3)

display(plt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probability Distribution

# COMMAND ----------

# Convert to Pandas for visualization
prob_dist_pd = reporting_df.select("positive_probability", "prediction").toPandas()

# Create histogram of probabilities
plt.figure(figsize=(12, 6))
for pred, group in prob_dist_pd.groupby("prediction"):
    sns.histplot(group["positive_probability"], bins=20, alpha=0.5, label=f"Prediction {pred}")

plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.3)

display(plt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Reports

# COMMAND ----------

# Prepare summary for export
summary_report = {
    "report_generated": datetime.now().isoformat(),
    "total_records": reporting_df.count(),
    "time_period": time_period,
    "prediction_distribution": summary_stats.toPandas().to_dict(orient="records"),
    "category_distribution": category_analysis.toPandas().to_dict(orient="records")
}

# Convert to JSON
import json
summary_json = json.dumps(summary_report)

# Write to a Delta table
spark.createDataFrame([(
    datetime.now().isoformat(),
    time_period,
    reporting_df.count(),
    summary_json
)], ["report_date", "time_period", "record_count", "summary"]) \
    .write.format("delta").mode("append").saveAsTable("prediction_reports")

print("Report saved to 'prediction_reports' table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we have:
# MAGIC 1. Loaded prediction data
# MAGIC 2. Generated summary statistics
# MAGIC 3. Created time series analysis
# MAGIC 4. Analyzed predictions by category
# MAGIC 5. Examined value and probability distributions
# MAGIC 6. Saved reports for future reference
# MAGIC 
# MAGIC These reports provide insights into model performance and prediction patterns. 