# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Processing Notebook
# MAGIC 
# MAGIC This notebook demonstrates basic data processing operations in Databricks using Apache Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Setup

# COMMAND ----------

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

# Example: Reading data from a CSV file
file_path = "/mnt/data/sample_data.csv"

# Define schema for better performance
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("date", DateType(), True),
    StructField("value", DoubleType(), True),
    StructField("category", StringType(), True)
])

# Read the CSV file with the defined schema
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .schema(schema) \
    .load(file_path)

# Display the first few rows of the dataframe
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration

# COMMAND ----------

# Get basic statistics of the dataframe
display(df.describe())

# Count the number of rows
print(f"Total number of records: {df.count()}")

# Check for null values
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation

# COMMAND ----------

# Example transformations
transformed_df = df \
    .withColumn("year", year(col("date"))) \
    .withColumn("month", month(col("date"))) \
    .withColumn("value_normalized", col("value") / lit(100)) \
    .withColumn("category_upper", upper(col("category")))

# Display the transformed dataframe
display(transformed_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Aggregation

# COMMAND ----------

# Group by category and calculate aggregates
agg_df = df.groupBy("category") \
    .agg(
        count("*").alias("count"),
        avg("value").alias("avg_value"),
        min("value").alias("min_value"),
        max("value").alias("max_value"),
        sum("value").alias("total_value")
    ) \
    .orderBy(desc("count"))

display(agg_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

# Convert Spark DataFrame to Pandas for visualization
pandas_df = df.toPandas()

# Create a simple visualization
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=pandas_df)
plt.title('Value Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Value')
plt.xticks(rotation=45)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Storage

# COMMAND ----------

# Save the transformed data as a Delta table
transformed_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("processed_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Perform some simple data quality checks
quality_checks = {
    "Total Records": df.count(),
    "Null Values": df.filter(col("id").isNull() | col("value").isNull()).count(),
    "Negative Values": df.filter(col("value") < 0).count(),
    "Categories": df.select("category").distinct().count()
}

for check_name, check_value in quality_checks.items():
    print(f"{check_name}: {check_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we've demonstrated:
# MAGIC 1. Data ingestion from CSV
# MAGIC 2. Basic data exploration
# MAGIC 3. Data transformation with Spark SQL functions
# MAGIC 4. Aggregation and summary statistics
# MAGIC 5. Basic visualization
# MAGIC 6. Saving data as a Delta table
# MAGIC 7. Simple data quality checks 