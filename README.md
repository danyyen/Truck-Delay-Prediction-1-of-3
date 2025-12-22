# **Build End-to-End ML Pipeline for Truck Delay Classification**


The project addresses a critical challenge faced by the logistics industry. Delayed truck shipments not only result in increased operational costs but also impact customer satisfaction. Timely delivery of goods is essential to meet customer expectations and maintain the competitiveness of logistics companies.
By accurately predicting truck delays, logistics companies can:
* Improve operational efficiency by allocating resources more effectively.
* Enhance customer satisfaction by providing more reliable delivery schedules.
* Optimize route planning to reduce delays caused by traffic or adverse weather conditions.
* Reduce costs associated with delayed shipments, such as penalties or compensation to customers.

# Aim
The primary objective of this project is to create an end-to-end machine learning pipeline for truck delay classification. This pipeline will encompass data fetching, creating a feature store, data preprocessing, and feature engineering.

# Tech Stack
➔ Language: Python, SQL
➔ Libraries: NumPy, Pandas, PyMySQL , Psycopg2, Matplotlib, Seaborn
➔ Data Storage: PostgreSQL,MySQL, AWS RDS, Hopsworks
➔ Data Visual Tool(SQL): MySQL Workbench, Pgadmin4
➔ Feature Store: Hopsworks
➔ Cloud Platform: AWS Sagemaker

# Data Description
The project involves the following data tables:
*  City Weather: Weather data for various cities
*  Routes: Information about truck routes, including origin, destination, distance, and travel time
*  Drivers: Details about truck drivers, including names and experience
*  Routes Weather: Weather conditions specific to each route
*  Trucks: Information about the trucks used in logistics operations
*  Traffic: Traffic-related data
*  Truck Schedule: Schedules and timing information for trucks

# ** This project is the first part of a three-part series ** # 
This is aimed at solving the truck delay prediction problem. In this initial phase, we will utilize PostgreSQL and MYSQL in AWS Redshift to store the data, perform data retrieval, and conduct basic exploratory data analysis (EDA). With Hopsworks feature store, we will build a pipeline that includes data processing feature engineering and prepare the data for model building.

![image.png](https://images.pexels.com/photos/2199293/pexels-photo-2199293.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)



## **Approach**


* Introduction to End-to-End Pipelines:
  * Understanding the fundamental concepts and importance of end-to-end pipelines


* Database Setup:
  * Creating AWS RDS instances for MySQL and PostgreSQL
  * Setting up MySQL Workbench and pgAdmin4 for database management


* Data Analysis:
  * Performing data analysis using SQL on MySQL Workbench and pgAdmin4


* AWS SageMaker Setup


* Exploratory Data Analysis (EDA):
  * Conducting exploratory data analysis to understand essential features and the dataset's characteristics


* Feature Store:
  * Understanding the concept of a feature store and its significance in machine learning projects
  * Understanding how Hopsworks works to facilitate project creation and feature group management


* Data Retrieval from Feature Stores

* Fetching data from feature stores for further analysis


* Data Preprocessing and Feature Engineering


* Data Storage:
  * Storing the final engineered features in the feature store for easy access and consistency

