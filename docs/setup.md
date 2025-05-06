# Databricks Project Setup Guide

This document provides step-by-step instructions for setting up the Databricks project environment.

## Prerequisites

Before getting started, ensure you have the following:

- Access to a Databricks workspace (AWS, Azure, or GCP)
- Admin or contributor permissions on the workspace
- Access to data sources (S3, Azure Blob Storage, databases, etc.)
- Git installed on your local machine (for repo management)
- Python 3.8+ installed locally (for development)

## Setting Up Your Environment

### 1. Create a Databricks Workspace

If you don't already have a Databricks workspace:

#### Azure Databricks
1. Log in to the Azure Portal
2. Create a new Azure Databricks service
3. Select appropriate region and pricing tier (Premium recommended for MLflow and Delta Lake features)
4. Launch the workspace after deployment

#### AWS Databricks
1. Sign up for Databricks on AWS
2. Create a new workspace through the Databricks console
3. Configure cross-account IAM role for AWS integration
4. Launch the workspace

### 2. Set Up Cluster

1. In your Databricks workspace, go to "Compute" in the left sidebar
2. Click "Create Cluster"
3. Configure the cluster:
   - Provide a meaningful cluster name
   - Select Databricks Runtime (DBR) version: 10.4 ML or later
   - Choose appropriate node type (memory-optimized for data processing, GPU-enabled for deep learning)
   - Set autoscaling parameters
   - Configure auto-termination
4. Click "Create Cluster"

### 3. Clone the Repository

#### Option 1: Using Databricks Repos (Recommended)
1. In your Databricks workspace, go to "Repos" in the left sidebar
2. Click "Add Repo"
3. Enter the Git repository URL
4. Click "Create"

#### Option 2: Manual Upload
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your-organization/databricks-project.git
   ```
2. Upload the files to your Databricks workspace using the UI or the Databricks CLI

### 4. Set Up Secret Management

To securely manage credentials for data sources:

1. Set up a Databricks secret scope:
   ```
   databricks secrets create-scope --scope aws
   ```

2. Add secrets to the scope:
   ```
   databricks secrets put --scope aws --key access_key
   databricks secrets put --scope aws --key secret_key
   ```

3. For database credentials:
   ```
   databricks secrets create-scope --scope db-credentials
   databricks secrets put --scope db-credentials --key username
   databricks secrets put --scope db-credentials --key password
   ```

### 5. Mount Storage

To make your data easily accessible, mount cloud storage:

1. Create a new notebook
2. Run the following code to mount AWS S3:
   ```python
   dbutils.fs.mount(
     source = "s3a://your-bucket-name",
     mount_point = "/mnt/data",
     extra_configs = {
       "fs.s3a.access.key": dbutils.secrets.get(scope="aws", key="access_key"),
       "fs.s3a.secret.key": dbutils.secrets.get(scope="aws", key="secret_key")
     }
   )
   ```

3. For Azure Blob Storage:
   ```python
   dbutils.fs.mount(
     source = "wasbs://container@account.blob.core.windows.net",
     mount_point = "/mnt/data",
     extra_configs = {
       "fs.azure.account.key.account.blob.core.windows.net": dbutils.secrets.get(scope="azure", key="storage_key")
     }
   )
   ```

### 6. Install Required Libraries

1. Go to your cluster settings
2. Select the "Libraries" tab
3. Click "Install New"
4. Choose "PyPI" as the source
5. Enter the package name or upload a requirements.txt file
6. Commonly required packages:
   - delta
   - mlflow
   - scikit-learn
   - matplotlib
   - seaborn
   - pandas
   - numpy
   - pytest

### 7. Set Up Delta Tables

Run the data ingestion notebook to create Delta tables:

1. Navigate to the `notebooks/` directory
2. Open `data_ingestion.py`
3. Set the appropriate parameters for your data source
4. Run the notebook to create the initial Delta tables

### 8. Configure a Job

Set up a scheduled job to run the pipeline:

1. Go to "Jobs" in the left sidebar
2. Click "Create Job"
3. Add a task:
   - Select the `data_ingestion.py` notebook
   - Choose a cluster (existing or create a new job cluster)
   - Set parameters as needed
4. Add subsequent tasks with dependencies:
   - data_processing.py
   - machine_learning.py
   - batch_scoring.py
   - reporting.py
5. Set the schedule for automatic runs
6. Save the job

## Testing Your Setup

To verify your setup is working correctly:

1. Run the data ingestion notebook manually
2. Check that data has been loaded into Delta tables
3. Run the data processing notebook
4. Verify processed tables are created
5. Run the machine learning notebook
6. Check that the model is trained and registered in MLflow
7. Run the batch scoring notebook
8. Verify predictions are generated
9. Run the reporting notebook
10. Check that visualizations are displayed

## Troubleshooting

### Common Issues:

#### Mount Error
If you encounter issues mounting storage:
- Verify secret values are correct
- Check permissions on the storage account/bucket
- Ensure network connectivity between Databricks and storage

#### Package Installation Errors
If libraries fail to install:
- Check for version conflicts
- Try installing packages one by one
- Use init scripts for complex installations

#### Job Failure
If jobs fail to run:
- Check the cluster is running and has sufficient resources
- Verify data sources are accessible
- Look at the job run logs for specific error messages

## Next Steps

After setting up your environment:

1. Review the [project structure documentation](project_structure.md)
2. Follow the [user guide](user_guide.md) for specific usage instructions
3. Customize the notebooks and configurations for your specific use case

## Support

If you encounter issues:
- Check the Databricks documentation at [docs.databricks.com](https://docs.databricks.com/)
- Consult the project wiki for project-specific guidance
- Contact the project maintainers 