!pip install psycopg2==2.9.7

# Import the psycopg2 library for PostgreSQL connection
import psycopg2

# Import the pandas library for data manipulation
import pandas as pd

# Establish a connection to the PostgreSQL database
postgres_connection = psycopg2.connect(
    user="postgres",             # PostgreSQL username
    password="your_passowrd",    # Password for the database
    host="host_id.rds.amazonaws.com",   # Host ID of the RDS instance
    database="DB",               # Name of the database
    port="5432"                  # Port number for PostgreSQL
)


# Read data from the "routes_details" table in the PostgreSQL database
routes_df = pd.read_sql("Select * from routes_details", postgres_connection)

# Display the first few rows of the routes dataframe
routes_df.head()


# Read data from the "routes_weather" table in the PostgreSQL database
route_weather = pd.read_sql("Select * from routes_weather", postgres_connection)

# Display the first few rows of the route weather dataframe
route_weather.head()


# Rename the column for consistency
route_weather=route_weather.rename(columns={'Date':'date'})

!pip install pymysql==1.1.0

# Import the pymysql library for MySQL connection
import pymysql

# Import the numpy library and alias it as np
import numpy as np

# Establish a connection to the MySQL database
mysql_connection = pymysql.connect(
     host = "host_id.rds.amazonaws.com",  # Host ID of the RDS instance
     user = "admin",                       # MySQL username
     password = "your_password",           # Password for the database
     database = "DB"                       # Name of the database
)


# Read data from the "drivers_details" table in the MySQL database
drivers_df = pd.read_sql("Select * from drivers_details", mysql_connection)

# Display the first two rows of the drivers dataframe
drivers_df.head(2)


# Read data from the "truck_details" table in the MySQL database
trucks_df = pd.read_sql("Select * from truck_details", mysql_connection)

# Display the first few rows of the trucks dataframe
trucks_df.head()


# Read data from the "traffic_details" table in the MySQL database
traffic_df = pd.read_sql("Select * from traffic_details", mysql_connection)

# Display the first few rows of the traffic dataframe
traffic_df.head()


# Read data from the "truck_schedule_data" table in the MySQL database
schedule_df = pd.read_sql("Select * from truck_schedule_data", mysql_connection)

# Display the first few rows of the schedule dataframe
schedule_df.head()


# Read data from the "city_weather" table in the MySQL database
weather_df = pd.read_sql("Select * from city_weather", mysql_connection)

# Display the first few rows of the weather dataframe
weather_df.head()


# Import Libraries

# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

# Change dates to datetime
weather_df['date'] = pd.to_datetime(weather_df['date'])
route_weather['date'] = pd.to_datetime(route_weather['date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
schedule_df['departure_date'] = pd.to_datetime(schedule_df['departure_date'])
schedule_df['estimated_arrival'] = pd.to_datetime(schedule_df['estimated_arrival'])
route_weather['date'] = pd.to_datetime(route_weather['date'])

# Driver's data
drivers_df.head(2)

# Driver's data info
drivers_df.info()

# statistics of various columns
drivers_df.describe(include='all')

# List of numerical columns to visualize
drivers_num_cols = ['age', 'experience', 'ratings', 'average_speed_mph']

# Loop through each numerical column and create histograms with KDE
for col in drivers_num_cols:
    plt.figure(figsize=(10, 5))

    # Create a histogram with KDE using seaborn
    sns.histplot(drivers_df[col], bins=30, kde=True)
    # Set the title
    plt.title(f'{col} distribution')
    # Set the label for the x-axis
    plt.xlabel(f'{col}')
    plt.show()

# Counts of gender
drivers_df['gender'].value_counts()

# Value counts of driving style
drivers_df['driving_style'].value_counts()

# Setting figure size
plt.figure(figsize=(10, 5))
# plotting scatter plot between ratings and average speed
sns.scatterplot(x='ratings', y='average_speed_mph', data=drivers_df)
plt.title('Ratings vs. Average Speed')
plt.xlabel('Ratings (out of 10)')
plt.ylabel('Average Speed (mph)')
plt.show()

# Boxplot between gender and ratings
sns.boxplot(x='gender', y='ratings', data=drivers_df, palette='Set2')
plt.title('Driver Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Ratings (out of 10)')
plt.show()

# Truck data head
trucks_df.head()

# Info
trucks_df.info()

# statistics of various columns
trucks_df.describe(include='all')

# Numerical cols in truck's dataset
truck_num_cols = ['truck_age', 'load_capacity_pounds', 'mileage_mpg']

# plotting histogram for each column
for col in truck_num_cols:
  plt.figure(figsize=(10, 5))
  sns.histplot(trucks_df[col], bins=30, kde=True)
  plt.title(f'{col} distribution')
  plt.xlabel(f'{col}')
  plt.show()

# According to the histogram, setting low mileage to be 15
low_mileage_threshold = 15

# Filter trucks with low mileage
low_mileage_trucks = trucks_df[trucks_df['mileage_mpg'] <= low_mileage_threshold]

# overview of data of low mileage trucks
low_mileage_trucks.head()

# Age distribution of low mileage trucks
plt.figure(figsize=(10, 5))
sns.histplot(low_mileage_trucks['truck_age'], bins=30, kde=True)
plt.title(f"Low Mileage Truck's Age distribution")
plt.xlabel("Age")
plt.show()

# Display the first rows
routes_df.head()

# Information on dataframe
routes_df.info()

# Traffic data head
traffic_df.head()

# Info
traffic_df.info()

# Sum of null values
traffic_df.isnull().sum()

# statistical description
traffic_df.describe()

# Sum of null values
traffic_df.isnull().sum()

# statistical description
traffic_df.describe()

def categorize_time(hour):
    """
    Categorizes hours of the day into time periods.

    Args:
    hour (int): Hour in 24-hour format.

    Returns:
    str: Categorized time period.
    """
    if 300 <= hour < 600:
        return 'Early Morning'
    elif 600 <= hour < 1200:
        return 'Morning'
    elif 1200 <= hour < 1600:
        return 'Noon'
    elif 1600 <= hour < 2000:
        return 'Evening'
    elif 2000 <= hour < 2300:
        return 'Night'
    elif 0 <= hour < 300:
        return 'Night'

# Create a copy of traffic_df
traffic = traffic_df.copy()

# Apply the categorize_time function to create a new column 'time_category'
traffic['time_category'] = traffic['hour'].apply(categorize_time)

# Group by 'time_category' and calculate the mean of 'no_of_vehicles'
mean_vehicles_by_time = traffic.groupby('time_category')['no_of_vehicles'].mean()


# print
mean_vehicles_by_time

!pip install -U hopsworks==3.2.0

# Import the necessary library
import hopsworks

# Log in to the Hopsworks project
project = hopsworks.login()

# Get the feature store associated with the project
fs = project.get_feature_store()


# Display the first two rows of the drivers DataFrame
drivers_df.head(2)

# Display information about the drivers DataFrame (e.g., column names, data types)
drivers_df.info()

drivers_df['event_time'] = pd.to_datetime('2023-08-23')

drivers_df.isna().sum()

# Filling the null values with Unknown
drivers_df['driving_style']=drivers_df['driving_style'].fillna('Unknown')
drivers_df['gender']=drivers_df['gender'].fillna('Unknown')

drivers_df.columns

# Create feature group for drivers details
drivers_fg = fs.get_or_create_feature_group(
    name="drivers_details_fg",                # Name of the feature group
    version=1,                                # Version number
    description="Drivers data",               # Description of the feature group
    primary_key=['driver_id'],                # Primary key(s) for the feature group
    event_time='event_time',                  # Event time column
    online_enabled=False                      # Online feature store capability
)

# Insert the drivers DataFrame into the feature group
drivers_fg.insert(drivers_df)

# Sort values
drivers_df=drivers_df.sort_values(["event_time","driver_id"])

# List of feature descriptions for drivers
feature_descriptions_drivers = [

    {"name": "driver_id", "description": "unique identification for each driver"},
    {"name": "name", "description": "name of the truck driver"},
    {"name": "gender", "description": "gender of the truck driver"},
    {"name": "age", "description": "age of the truck driver"},
    {"name": "experience", "description": "experience of the truck driver in years"},
    {"name": "driving_style", "description": "driving style of the truck driver, conservative or proactive"},
    {"name": "ratings", "description": "average rating of the truck driver on a scale of 1 to 5"},
    {"name": "vehicle_no", "description": "the number of the driver’s truck"},
    {"name": "average_speed_mph", "description": "average speed of the truck driver in miles per hour"},
    {"name": "event_time", "description": "dummy event time"}

]

# Iterate through the feature descriptions and update them in the feature group
for desc in feature_descriptions_drivers:
    drivers_fg.update_feature_description(desc["name"], desc["description"])


# Configure statistics for the feature group
drivers_fg.statistics_config = {
    "enabled": True,        # Enable statistics calculation
    "histograms": True,     # Include histograms in the statistics
    "correlations": True    # Include correlations in the statistics
}

# Update the statistics configuration for the feature group
drivers_fg.update_statistics_config()

# Compute statistics for the feature group
drivers_fg.compute_statistics()


# Displaying head of the data
trucks_df.head()

# Displaying information
trucks_df.info()

# Sum of null values
trucks_df.isna().sum()

trucks_df['fuel_type'].unique()

# Filling the null values with Unknown
trucks_df['fuel_type']=trucks_df['fuel_type'].replace("",'Unknown')



trucks_df['fuel_type'].value_counts()

trucks_df['event_time'] = pd.to_datetime('2023-08-23')

trucks_df=trucks_df.sort_values(["event_time","truck_id"])

# Create a feature group for truck details
truck_fg = fs.get_or_create_feature_group(
    name="truck_details_fg",          # Name of the feature group
    version=1,                        # Version number
    description="Truck data",         # Description of the feature group
    primary_key=['truck_id'],         # Primary key(s) for the feature group
    event_time='event_time',          # Event time column
    online_enabled=False              # Online feature store capability (set to False)
)


truck_fg.insert(trucks_df)

# Add feature descriptions

feature_descriptions_trucks = [
    {"name":'truck_id',"description":"the unique identification number of the truck"},
    {"name":'truck_age',"description":"age of the truck in years"},
    {"name":'load_capacity_pounds',"description":"loading capacity of the truck in years"},
    {"name":'mileage_mpg',"description": "mileage of the truck in miles per gallon"},
    {"name":'fuel_type',"description":"fuel type of the truck"},
    {"name": "event_time", "description": "dummy event time"}

]

for desc in feature_descriptions_trucks:
    truck_fg.update_feature_description(desc["name"], desc["description"])

truck_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

truck_fg.update_statistics_config()
truck_fg.compute_statistics()

# Display the head
routes_df.head()

# Routes Information
routes_df.info()

# Sum of null values
routes_df.isna().sum()

routes_df['event_time'] = pd.to_datetime('2023-08-23')

routes_df=routes_df.sort_values(["event_time","route_id"])

# Create feature group for route details
routes_fg = fs.get_or_create_feature_group(
    name="routes_details_fg",         # Name of the feature group
    version=1,                        # Version number
    description="Routes data",        # Description of the feature group
    primary_key=['route_id'],         # Primary key(s) for the feature group
    event_time='event_time',          # Event time column
    online_enabled=False              # Online feature store capability (set to False)
)


routes_fg.insert(routes_df)

# Add feature descriptions

feature_descriptions_routes = [
    {"name": 'route_id', "description": "the unique identifier of the routes"},
    {"name": 'origin_id', "description": "the city identification number for the origin city"},
    {"name": 'destination_id', "description": " the city identification number for the destination"},
    {"name": 'distance', "description": " the distance between the origin and destination cities in miles"},
    {"name": 'average_hours', "description": "average time needed to travel from the origin to the destination in hours"},
    {"name": "event_time", "description": "dummy event time"}

]

for desc in feature_descriptions_routes:
    routes_fg.update_feature_description(desc["name"], desc["description"])

routes_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

routes_fg.update_statistics_config()
routes_fg.compute_statistics()

# Display the head
schedule_df.head()

# Display data information
schedule_df.info()

# Sum of null values
schedule_df.isna().sum()

# sorting
schedule_df=schedule_df.sort_values(["estimated_arrival","truck_id"])

# Create  feature group for truck schedule details
truck_schedule_fg = fs.get_or_create_feature_group(
    name="truck_schedule_details_fg",  # Name of the feature group
    version=1,                          # Version number
    description="Truck Schedule data",  # Description of the feature group
    primary_key=['truck_id','route_id'], # Primary key(s) for the feature group
    event_time='estimated_arrival',     # Event time column
    online_enabled=True                  # Online feature store capability (set to True)
)


truck_schedule_fg.insert(schedule_df)

# Add feature descriptions
feature_descriptions_schedule = [
    {"name": 'truck_id', "description": "the unique identifier of the truck"},
    {"name": 'route_id', "description": "the unique identifier of the route"},
    {"name": 'departure_date', "description": "departure DateTime of the truck"},
    {"name": 'estimated_arrival', "description": "estimated arrival DateTime of the truck"},
    {"name": 'delay', "description": "binary variable if the truck’s arrival was delayed, 0 for on-time arrival and 1 for delayed arrival"},
]

for desc in feature_descriptions_schedule:
    truck_schedule_fg.update_feature_description(desc["name"], desc["description"])

truck_schedule_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

truck_schedule_fg.update_statistics_config()
truck_schedule_fg.compute_statistics()

traffic_df.head()

traffic_df.info()

traffic_df.isna().sum()

traffic_df=traffic_df.sort_values(['date','route_id','hour'])

traffic_fg = fs.get_or_create_feature_group(
    name="traffic_details_fg",
    version=1,
    description="Traffic data",
    primary_key=['route_id','hour'],
    event_time='date',
    online_enabled=True
)

traffic_fg.insert(traffic_df)

feature_descriptions_traffic = [
     {"name": 'route_id', "description": "the identification number of the route"},
     {"name": 'date', "description": " date of the traffic observation"},
     {"name": 'hour', "description": "the hour of the observation as a number in 24-hour format"},
     {"name": 'no_of_vehicles', "description": "the number of vehicles observed on the route"},
     {"name": 'accident', "description": "binary variable to denote if an accident was observed"}

]

for desc in feature_descriptions_traffic:
    traffic_fg.update_feature_description(desc["name"], desc["description"])

traffic_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

traffic_fg.update_statistics_config()
traffic_fg.compute_statistics()

weather_df.head()

weather_df.info()

weather_df.isna().sum()

weather_df=weather_df.sort_values(['date','city_id','hour'])

city_weather_fg = fs.get_or_create_feature_group(
    name="city_weather_details_fg",
    version=1,
    description="City Weather data",
    primary_key=['city_id','hour'],
    event_time='date',
    online_enabled=True
)

city_weather_fg.insert(weather_df)

feature_descriptions_weather = [
    {"name": 'city_id', "description":  'the unique identifier of the city'},
    {"name": 'date', "description":  'date of the observation'},
    {"name": 'hour', "description": 'the hour of the observation as a number in 24hour format'},
    {"name": 'temp', "description":  'temperature in Fahrenheit'},
    {"name": 'wind_speed', "description":  'wind speed in miles per hour'},
    {"name": 'description', "description":  'description of the weather conditions such as Clear, Cloudy, etc'},
    {"name": 'precip', "description":  'precipitation in inches'},
    {"name": 'humidity', "description":  'humidity observed'},
    {"name": 'visibility', "description":  'visibility observed in miles per hour'},
    {"name": 'pressure', "description":  'pressure observed in millibar'},
    {"name": 'chanceofrain', "description":  'chances of rain'},
    {"name": 'chanceoffog', "description":  'chances of fog'},
    {"name": 'chanceofsnow', "description":  'chances of snow'},
    {"name": 'chanceofthunder', "description":  'chances of thunder'}

]

for desc in feature_descriptions_weather:
    city_weather_fg.update_feature_description(desc["name"], desc["description"])

city_weather_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

city_weather_fg.update_statistics_config()
city_weather_fg.compute_statistics()

route_weather.head()

route_weather.info()

route_weather.isna().sum()

route_weather=route_weather.sort_values(by=['date','route_id'])

route_weather_fg = fs.get_or_create_feature_group(
    name="route_weather_details_fg",
    version=1,
    description="Route Weather data",
    primary_key=['route_id'],
    event_time='date',
    online_enabled=True
)

route_weather_fg.insert(route_weather)

feature_descriptions_route_weather = [

    {"name": 'route_id', "description":  'the unique identifier of the city'},
    {"name": 'date', "description":  'date of the observation'},
    {"name": 'temp', "description":  'temperature in Fahrenheit'},
    {"name": 'wind_speed', "description":  'wind speed in miles per hour'},
    {"name": 'description', "description":  'description of the weather conditions such as Clear, Cloudy, etc'},
    {"name": 'precip', "description":  'precipitation in inches'},
    {"name": 'humidity', "description":  'humidity observed'},
    {"name": 'visibility', "description":  'visibility observed in miles per hour'},
    {"name": 'pressure', "description":  'pressure observed in millibar'},
    {"name": 'chanceofrain', "description":  'chances of rain'},
    {"name": 'chanceoffog', "description":  'chances of fog'},
    {"name": 'chanceofsnow', "description":  'chances of snow'},
    {"name": 'chanceofthunder', "description":  'chances of thunder'}

]

for desc in feature_descriptions_route_weather:
    route_weather_fg.update_feature_description(desc["name"], desc["description"])

route_weather_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

route_weather_fg.update_statistics_config()
route_weather_fg.compute_statistics()

routes_df_fg = fs.get_feature_group('routes_details_fg', version=1)
query = routes_df_fg.select_all()
routes_df=query.read()

route_weather_fg = fs.get_feature_group('route_weather_details_fg', version=1)
query = route_weather_fg.select_all()
route_weather=query.read()

drivers_df_fg = fs.get_feature_group('drivers_details_fg', version=1)
query = drivers_df_fg.select_all()
drivers_df=query.read()

trucks_df_fg = fs.get_feature_group('truck_details_fg', version=1)
query = trucks_df_fg.select_all()
trucks_df=query.read()

traffic_df_fg = fs.get_feature_group('traffic_details_fg', version=1)
query = traffic_df_fg.select_all()
traffic_df=query.read()

schedule_df_fg = fs.get_feature_group('truck_schedule_details_fg', version=1)
query = schedule_df_fg.select_all()
schedule_df=query.read()

weather_df_fg = fs.get_feature_group('city_weather_details_fg', version=1)
query = weather_df_fg.select_all()
weather_df=query.read()

drivers_df.head(2)

drivers_df=drivers_df.drop(columns=['event_time'])

# Check the null values
drivers_df.isna().sum()

# Duplicates in drivers data
drivers_df[drivers_df.duplicated(subset=['driver_id'])]

# Trucks data
trucks_df.head(2)

trucks_df=trucks_df.drop(columns=['event_time'])

# Check null values
trucks_df.isna().sum()

# Checking the different load capacities
trucks_df['load_capacity_pounds'].unique()

# Most common value
trucks_df['load_capacity_pounds'].mode()

#check null values
trucks_df.isna().sum()

# Check for duplicates
trucks_df[trucks_df.duplicated(subset=['truck_id'])]

#
routes_df.head(2)

routes_df=routes_df.drop(columns=['event_time'])

# Sum of null values
routes_df.isna().sum()

# check duplicates
routes_df[routes_df.duplicated(subset=['route_id'])]

# check duplicates across origin and destination
routes_df[routes_df.duplicated(subset=['route_id','destination_id','origin_id'])]

schedule_df.head(2)

# sum of null values in schedule
schedule_df.isna().sum()

# check for duplicates
schedule_df[schedule_df.duplicated()]

weather_df.head(2)

# statistical description
weather_df.describe()

# check for duplicates
weather_df[weather_df.duplicated(subset=['city_id','date','hour'])]

# drop duplicates
weather_df=weather_df.drop_duplicates(subset=['city_id','date','hour'])

# drop unnecessary cols
weather_df=weather_df.drop(columns=['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'])

# Convert 'hour' to a 4-digit string format
weather_df['hour'] = weather_df['hour'].apply(lambda x: f'{x:04d}')

# Convert 'hour' to datetime format
weather_df['hour'] = pd.to_datetime(weather_df['hour'], format='%H%M').dt.time

# Combine 'date' and 'hour' to create a new datetime column 'custom_date' and insert it at index 1
weather_date_val = pd.to_datetime(weather_df['date'].astype(str) + ' ' + weather_df['hour'].astype(str))
weather_df.insert(1, 'custom_date', weather_date_val)


weather_df.head(2)

weather_df.describe()

#drop city_id from here
route_weather.head(2)

route_weather.describe()

# check for duplicates
route_weather[route_weather.duplicated(subset=['route_id','date'])]

# Drop unnecessary cols
route_weather=route_weather.drop(columns=['chanceofrain','chanceoffog','chanceofsnow','chanceofthunder'])

route_weather.isna().sum()

traffic_df.head(2)

traffic_df[traffic_df.duplicated(subset=['route_id','date','hour'])]

traffic_df=traffic_df.drop_duplicates(subset=['route_id','date','hour'],keep='first')

traffic_df.isna().sum()

# Convert 'hour' to a 4-digit string format
traffic_df['hour'] = traffic_df['hour'].apply(lambda x: f'{x:04d}')

# Convert 'hour' to datetime format
traffic_df['hour'] = pd.to_datetime(traffic_df['hour'], format='%H%M').dt.time

# Combine 'date' and 'hour' to create a new datetime column 'custom_date' and insert it at index 1
traffic_custom_date = pd.to_datetime(traffic_df['date'].astype(str) + ' ' + traffic_df['hour'].astype(str))
traffic_df.insert(1, 'custom_date', traffic_custom_date)


traffic_df.head(5)

schedule_df.head(2)

schedule_df.isna().sum()

schedule_df.describe(include='all')

schedule_df[schedule_df.duplicated(subset=['truck_id','route_id','departure_date'])]

schedule_df.insert(0,'unique_id',np.arange(len(schedule_df)))

nearest_6h_schedule_df=schedule_df.copy()

nearest_6h_schedule_df['estimated_arrival']=nearest_6h_schedule_df['estimated_arrival'].dt.ceil("6H")
nearest_6h_schedule_df['departure_date']=nearest_6h_schedule_df['departure_date'].dt.floor("6H")

nearest_6h_schedule_df.head(2)


# Assign a new column 'date' using a list comprehension to generate date ranges between 'departure_date' and 'estimated_arrival' with a frequency of 6 hours
# This will create a list of date ranges for each row
# Explode the 'date' column to create separate rows for each date range

exploded_6h_scheduled_df=(nearest_6h_schedule_df.assign(date = [pd.date_range(start, end, freq='6H')
                      for start, end
                      in zip(nearest_6h_schedule_df['departure_date'], nearest_6h_schedule_df['estimated_arrival'])]).explode('date', ignore_index = True))

exploded_6h_scheduled_df.head(2)

schduled_weather=exploded_6h_scheduled_df.merge(route_weather,on=['route_id','date'],how='left')

schduled_weather.head(4)

# Define a custom function to calculate mode
def custom_mode(x):
    return x.mode().iloc[0]

# Group by specified columns and aggregate
schedule_weather_grp = schduled_weather.groupby(['unique_id','truck_id','route_id'], as_index=False).agg(
    route_avg_temp=('temp','mean'),
    route_avg_wind_speed=('wind_speed','mean'),
    route_avg_precip=('precip','mean'),
    route_avg_humidity=('humidity','mean'),
    route_avg_visibility=('visibility','mean'),
    route_avg_pressure=('pressure','mean'),
    route_description=('description', custom_mode)
)


schedule_weather_grp.head(2)

schedule_weather_merge=schedule_df.merge(schedule_weather_grp,on=['unique_id','truck_id','route_id'],how='left')

schedule_weather_merge.shape

schedule_weather_merge.isna().sum()

weather_df.head(2)

#take hourly as weather data available hourly
nearest_hour_schedule_df=schedule_df.copy()
nearest_hour_schedule_df['estimated_arrival_nearest_hour']=nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
nearest_hour_schedule_df['departure_date_nearest_hour']=nearest_hour_schedule_df['departure_date'].dt.round("H")
nearest_hour_schedule_route_df=pd.merge(nearest_hour_schedule_df, routes_df, on='route_id', how='left')

nearest_hour_schedule_route_df.shape

nearest_hour_schedule_route_df.dtypes

weather_df.dtypes

# Create a copy of the 'weather_df' DataFrame for manipulation
origin_weather_data = weather_df.copy()

# Drop the 'date' and 'hour' columns from 'origin_weather_data'
origin_weather_data = origin_weather_data.drop(columns=['date', 'hour'])

origin_weather_data.columns = ['origin_id','departure_date_nearest_hour', 'origin_temp', 'origin_wind_speed','origin_description', 'origin_precip',
       'origin_humidity', 'origin_visibility', 'origin_pressure']

# Create a copy of the 'weather_df' DataFrame for manipulation
destination_weather_data = weather_df.copy()

# Drop the 'date' and 'hour' columns from 'destination_weather_data'
destination_weather_data = destination_weather_data.drop(columns=['date', 'hour'])

destination_weather_data.columns = ['destination_id', 'estimated_arrival_nearest_hour','destination_temp', 'destination_wind_speed','destination_description', 'destination_precip',
       'destination_humidity', 'destination_visibility', 'destination_pressure' ]

# Merge 'nearest_hour_schedule_route_df' with 'origin_weather_data' based on specified columns
origin_weather_merge = pd.merge(nearest_hour_schedule_route_df, origin_weather_data, on=['origin_id','departure_date_nearest_hour'], how='left')

# Merge 'origin_weather_merge' with 'destination_weather_data' based on specified columns
origin_destination_weather = pd.merge(origin_weather_merge, destination_weather_data , on=['destination_id', 'estimated_arrival_nearest_hour'], how='left')


origin_destination_weather.head(2)

origin_destination_weather.shape

traffic_df.head(5)

traffic_df.dtypes

schedule_df.head(5)

schedule_df.dtypes

# Create a copy of the schedule DataFrame for manipulation
nearest_hour_schedule_df = schedule_df.copy()

# Round 'estimated_arrival' times to the nearest hour
nearest_hour_schedule_df['estimated_arrival'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")

# Round 'departure_date' times to the nearest hour
nearest_hour_schedule_df['departure_date'] = nearest_hour_schedule_df['departure_date'].dt.round("H")

nearest_hour_schedule_df.head(5)

hourly_exploded_scheduled_df=(nearest_hour_schedule_df.assign(custom_date = [pd.date_range(start, end, freq='H')  # Create custom date ranges
                      for start, end
                      in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])])  # Using departure and estimated arrival times
                      .explode('custom_date', ignore_index = True))  # Explode the DataFrame based on the custom date range

hourly_exploded_scheduled_df.head(10)

scheduled_traffic=hourly_exploded_scheduled_df.merge(traffic_df,on=['route_id','custom_date'],how='left')

# Define a custom aggregation function for accidents
def custom_agg(values):
    """
    Custom aggregation function to determine if any value in a group is 1 (indicating an accident).

    Args:
    values (iterable): Iterable of values in a group.

    Returns:
    int: 1 if any value is 1, else 0.
    """
    if any(values == 1):
        return 1
    else:
        return 0

# Group by 'unique_id', 'truck_id', and 'route_id', and apply custom aggregation
scheduled_route_traffic = scheduled_traffic.groupby(['unique_id', 'truck_id', 'route_id'], as_index=False).agg(
    avg_no_of_vehicles=('no_of_vehicles', 'mean'),
    accident=('accident', custom_agg)
)


scheduled_route_traffic.head(5)

origin_destination_weather_traffic_merge=origin_destination_weather.merge(scheduled_route_traffic,on=['unique_id','truck_id','route_id'],how='left')

origin_destination_weather_traffic_merge.head(5)

schedule_weather_merge.columns.intersection(origin_destination_weather_traffic_merge.columns)

merged_data_weather_traffic=pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['unique_id', 'truck_id', 'route_id', 'departure_date',
       'estimated_arrival', 'delay'], how='left')

merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df, on='truck_id', how='left')

# Merge merged_data with truck_data based on 'truck_id' column (Left Join)
final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df, left_on='truck_id', right_on = 'vehicle_no', how='left')

final_merge.shape

final_merge.head(5)

# Function to check if there is nighttime involved between arrival and departure time
def has_midnight(start, end):
    return int(start.date() != end.date())


# Apply the function to create a new column indicating nighttime involvement
final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)

final_merge[final_merge['is_midnight']==1]

fs_data = final_merge.sort_values(["estimated_arrival","unique_id"])

'''import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()'''

truck_eta_fg = fs.get_or_create_feature_group(
    name="final_data",
    version=1,
    description="Truck ETA Final Data",
    primary_key=['unique_id'],
    event_time='estimated_arrival',
    online_enabled=True,
)

fs_data.isna().sum()

fs_data.dtypes

fs_data['origin_description'] = fs_data['origin_description'].fillna("Unknown")

truck_eta_fg.insert(fs_data)

final_feature_descriptions = [
    {"name": 'unique_id', "description": "the unique identifier for each record"},
    {"name": 'truck_id', "description": "the unique identifier of the truck"},
    {"name": 'route_id', "description": "the unique identifier of the route"},
    {"name": 'departure_date', "description": "departure DateTime of the truck"},
    {"name": 'estimated_arrival', "description": "estimated arrival DateTime of the truck"},
    {"name": 'delay', "description": "binary variable if the truck’s arrival was delayed, 0 for on-time arrival and 1 for delayed arrival"},
    {"name": 'route_avg_temp', "description":  'Average temperature in Fahrenheit'},
    {"name": 'route_avg_wind_speed', "description":  'Average wind speed in miles per hour'},
    {"name": 'route_avg_precip', "description":  'Average precipitation in inches'},
    {"name": 'route_avg_humidity', "description":  'Average humidity observed'},
    {"name": 'route_avg_visibility', "description":  'Average visibility observed in miles per hour'},
    {"name": 'route_avg_pressure', "description":  'Average pressure observed in millibar'},
    {"name": 'route_description', "description":  'description of the weather conditions such as Clear, Cloudy, etc'},
    {"name": 'estimated_arrival_nearest_hour', "description":  'estimated arrival DateTime of the truck'},
    {"name": 'departure_date_nearest_hour', "description":  'departure DateTime of the truck'},
    {"name": 'origin_id', "description": "the city identification number for the origin city"},
    {"name": 'destination_id', "description": " the city identification number for the destination"},
    {"name": 'distance', "description": " the distance between the origin and destination cities in miles"},
    {"name": 'average_hours', "description": "average time needed to travel from the origin to the destination in hours"},
    {"name": 'origin_temp', "description":  'temperature in Fahrenheit'},
    {"name": 'origin_wind_speed', "description":  'wind speed in miles per hour'},
    {"name": 'origin_description', "description":  'description of the weather conditions such as Clear, Cloudy, etc'},
    {"name": 'origin_precip', "description":  'precipitation in inches'},
    {"name": 'origin_humidity', "description":  'humidity observed'},
    {"name": 'origin_visibility', "description":  'visibility observed in miles per hour'},
    {"name": 'origin_pressure', "description":  'pressure observed in millibar'},
    {"name": 'destination_temp', "description":  'temperature in Fahrenheit'},
    {"name": 'destination_wind_speed', "description":  'wind speed in miles per hour'},
    {"name": 'destination_description', "description":  'description of the weather conditions such as Clear, Cloudy, etc'},
    {"name": 'destination_precip', "description":  'precipitation in inches'},
    {"name": 'destination_humidity', "description":  'humidity observed'},
    {"name": 'destination_visibility', "description":  'visibility observed in miles per hour'},
    {"name": 'destination_pressure', "description":  'pressure observed in millibar'},
    {"name": 'avg_no_of_vehicles', "description": "the average number of vehicles observed on the route"},
    {"name": 'accident', "description": "binary variable to denote if an accident was observed"},
    {"name":'truck_age',"description":"age of the truck in years"},
    {"name":'load_capacity_pounds',"description":"loading capacity of the truck in years"},
    {"name":'mileage_mpg',"description": "mileage of the truck in miles per gallon"},
    {"name":'fuel_type',"description":"fuel type of the truck"},
    {"name": "driver_id", "description": "unique identification for each driver"},
    {"name": "name", "description": " name of the truck driver"},
    {"name": "gender", "description": "gender of the truck driver"},
    {"name": "age", "description": "age of the truck driver"},
    {"name": "experience", "description": " experience of the truck driver in years"},
    {"name": "driving_style", "description": "driving style of the truck driver, conservative or proactive"},
    {"name": "ratings", "description": "average rating of the truck driver on a scale of 1 to 5"},
    {"name": "vehicle_no", "description": "the number of the driver’s truck"},
    {"name": "average_speed_mph", "description": "average speed the truck driver in miles per hour"},
    {"name": 'is_midnight', "description": "binary variable to denote if it was midnight"}

]

for desc in final_feature_descriptions:
    truck_eta_fg.update_feature_description(desc["name"], desc["description"])

truck_eta_fg = fs.get_or_create_feature_group("final_data", version=1)
truck_eta_fg.statistics_config = {
    "enabled": True,
    "histograms": True,
    "correlations": True
}

truck_eta_fg.update_statistics_config()
truck_eta_fg.compute_statistics()
