# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC 
# MAGIC This notebook is responsible for ingesting data from various sources into the Databricks platform.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Parameters

# COMMAND ----------

# Define default parameters that can be overridden when the notebook is run
dbutils.widgets.text("data_source", "s3://example-bucket/raw-data/", "Data Source Path")
dbutils.widgets.text("target_table", "raw_data", "Target Table Name")
dbutils.widgets.dropdown("ingest_mode", "incremental", ["full", "incremental"], "Ingestion Mode")
dbutils.widgets.text("checkpoint_path", "/mnt/checkpoints/data_ingestion", "Checkpoint Path")

# Read parameters
data_source = dbutils.widgets.get("data_source")
target_table = dbutils.widgets.get("target_table")
ingest_mode = dbutils.widgets.get("ingest_mode")
checkpoint_path = dbutils.widgets.get("checkpoint_path")

print(f"Data Source: {data_source}")
print(f"Target Table: {target_table}")
print(f"Ingestion Mode: {ingest_mode}")
print(f"Checkpoint Path: {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Connection to Data Source

# COMMAND ----------

# Example: Set up S3 connection
spark.conf.set("spark.hadoop.fs.s3a.access.key", dbutils.secrets.get(scope="aws", key="access_key"))
spark.conf.set("spark.hadoop.fs.s3a.secret.key", dbutils.secrets.get(scope="aws", key="secret_key"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Schema for the Source Data

# COMMAND ----------

# Define schema for the incoming data for better performance
customer_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("registration_date", TimestampType(), True),
    StructField("last_activity_date", TimestampType(), True),
    StructField("account_status", StringType(), True),
    StructField("customer_segment", StringType(), True),
    StructField("lifetime_value", DoubleType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Data from Various Sources

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1: Ingest data from S3 (CSV files)

# COMMAND ----------

# Function to ingest data from CSV files
def ingest_from_csv(source_path, schema=None):
    """Ingest data from CSV files"""
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .option("delimiter", ",") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
        .schema(schema) \
        .load(source_path)
    
    return df

# Example usage
customer_data = ingest_from_csv(f"{data_source}/customers/", customer_schema)
display(customer_data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2: Ingest data from JDBC source (Database)

# COMMAND ----------

# Function to ingest data from JDBC source
def ingest_from_jdbc(jdbc_url, table_name, user, password, driver):
    """Ingest data from JDBC source"""
    df = spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", table_name) \
        .option("user", user) \
        .option("password", password) \
        .option("driver", driver) \
        .load()
    
    return df

# Example usage (uncomment to use)
# jdbc_url = "jdbc:postgresql://example-db.amazonaws.com:5432/database"
# table_name = "transactions"
# user = dbutils.secrets.get(scope="db-credentials", key="username")
# password = dbutils.secrets.get(scope="db-credentials", key="password")
# driver = "org.postgresql.Driver"
# 
# transaction_data = ingest_from_jdbc(jdbc_url, table_name, user, password, driver)
# display(transaction_data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3: Ingest data from JSON files

# COMMAND ----------

# Function to ingest data from JSON files
def ingest_from_json(source_path, schema=None):
    """Ingest data from JSON files"""
    df = spark.read.format("json") \
        .option("inferSchema", "false") \
        .schema(schema) \
        .load(source_path)
    
    return df

# Example usage
# Define schema for JSON data
product_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("description", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("category", StringType(), True),
    StructField("inventory", IntegerType(), True),
    StructField("last_updated", TimestampType(), True)
])

product_data = ingest_from_json(f"{data_source}/products/", product_schema)
display(product_data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 4: Streaming Data Ingestion

# COMMAND ----------

# Function to ingest streaming data
def ingest_streaming_data(source_path, schema, checkpoint_path):
    """Ingest streaming data"""
    df = spark.readStream.format("json") \
        .option("maxFilesPerTrigger", 1) \
        .option("path", source_path) \
        .schema(schema) \
        .load()
    
    # Add processing timestamp
    df = df.withColumn("processing_time", current_timestamp())
    
    # Write the streaming data to a Delta table
    query = df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", checkpoint_path) \
        .toTable(f"streaming_{target_table}")
    
    return query

# Example usage (uncomment to use)
# Order schema for streaming data
# order_schema = StructType([
#     StructField("order_id", StringType(), False),
#     StructField("customer_id", StringType(), True),
#     StructField("order_date", TimestampType(), True),
#     StructField("status", StringType(), True),
#     StructField("items", ArrayType(
#         StructType([
#             StructField("product_id", StringType(), True),
#             StructField("quantity", IntegerType(), True),
#             StructField("price", DoubleType(), True)
#         ])
#     ), True),
#     StructField("total_amount", DoubleType(), True)
# ])
# 
# streaming_query = ingest_streaming_data(
#     f"{data_source}/orders/stream/", 
#     order_schema, 
#     f"{checkpoint_path}/orders_stream"
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Function to perform basic data quality checks
def perform_data_quality_checks(df, primary_key_column):
    """Perform basic data quality checks"""
    total_count = df.count()
    
    # Check for duplicate primary keys
    duplicate_count = df.groupBy(primary_key_column).count().filter("count > 1").count()
    
    # Check for null values in primary key column
    null_pk_count = df.filter(col(primary_key_column).isNull()).count()
    
    # Count null values in each column
    null_counts = {column: df.filter(col(column).isNull()).count() for column in df.columns}
    
    # Print results
    print(f"Total records: {total_count}")
    print(f"Duplicate {primary_key_column} records: {duplicate_count}")
    print(f"Null values in {primary_key_column}: {null_pk_count}")
    print("Null counts by column:")
    for column, count in null_counts.items():
        print(f"  {column}: {count} ({count/total_count*100:.2f}%)")
    
    # Return validation status (True if all checks pass)
    return duplicate_count == 0 and null_pk_count == 0

# Run quality checks on customer data
customer_quality_check = perform_data_quality_checks(customer_data, "customer_id")
print(f"Customer data quality check passed: {customer_quality_check}")

# Run quality checks on product data
product_quality_check = perform_data_quality_checks(product_data, "product_id")
print(f"Product data quality check passed: {product_quality_check}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Data to Target Tables

# COMMAND ----------

# Function to save data to a Delta table
def save_to_delta_table(df, table_name, mode="overwrite"):
    """Save DataFrame to a Delta table"""
    df.write.format("delta") \
        .mode(mode) \
        .option("mergeSchema", "true") \
        .saveAsTable(table_name)
    
    print(f"Data saved to table: {table_name}")

# Save customer data
save_to_delta_table(customer_data, f"{target_table}_customers")

# Save product data
save_to_delta_table(product_data, f"{target_table}_products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incremental Load with Change Data Capture (CDC)

# COMMAND ----------

# Function to perform incremental loading using CDC principles
def incremental_load_with_cdc(source_df, target_table, key_columns, timestamp_col):
    """
    Perform incremental load with CDC
    
    source_df: The DataFrame containing the new/changed data
    target_table: The target Delta table name
    key_columns: List of columns that form the primary key
    timestamp_col: The column containing the last update timestamp
    """
    # Check if the target table exists
    table_exists = spark.catalog._jcatalog.tableExists(target_table)
    
    if not table_exists:
        # If the table doesn't exist, create it with the new data
        source_df.write.format("delta").saveAsTable(target_table)
        print(f"Created new table: {target_table}")
    else:
        # If the table exists, perform a merge operation
        from delta.tables import DeltaTable
        
        # Get the Delta table
        delta_table = DeltaTable.forName(spark, target_table)
        
        # Define the merge condition
        merge_condition = " AND ".join([f"tgt.{col} = src.{col}" for col in key_columns])
        
        # Perform the merge operation
        delta_table.alias("tgt").merge(
            source_df.alias("src"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=f"src.{timestamp_col} > tgt.{timestamp_col}"
        ).whenNotMatchedInsertAll().execute()
        
        print(f"Performed incremental merge on table: {target_table}")

# Example usage (assuming we're doing an incremental load of customer data)
if ingest_mode == "incremental":
    incremental_load_with_cdc(
        customer_data,
        f"{target_table}_customers_incremental",
        ["customer_id"],
        "last_activity_date"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record Ingestion Metrics

# COMMAND ----------

# Record ingestion metrics
ingestion_metrics = {
    "ingestion_time": datetime.datetime.now().isoformat(),
    "source": data_source,
    "target_table": target_table,
    "ingest_mode": ingest_mode,
    "customer_records": customer_data.count(),
    "product_records": product_data.count(),
    "customer_quality_check": customer_quality_check,
    "product_quality_check": product_quality_check
}

# Create a DataFrame from the metrics
metrics_df = spark.createDataFrame([ingestion_metrics])

# Save the metrics
metrics_df.write.format("delta") \
    .mode("append") \
    .saveAsTable("data_ingestion_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we have:
# MAGIC 1. Set up connections to data sources
# MAGIC 2. Defined schemas for structured ingestion
# MAGIC 3. Ingested data from multiple sources (CSV, JSON, JDBC, streaming)
# MAGIC 4. Performed basic data quality checks
# MAGIC 5. Saved the data to Delta tables
# MAGIC 6. Implemented incremental loading with CDC principles
# MAGIC 7. Recorded ingestion metrics for monitoring
# MAGIC 
# MAGIC Next steps: Proceed to data processing and transformation. 