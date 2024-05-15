# Sales Data Analysis with PySpark

This project involves analyzing sales data using PySpark. The analysis includes data cleaning, transformation, and visualization to derive meaningful insights from the sales data.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Data Analysis](#data-analysis)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to perform a comprehensive analysis of sales data using PySpark. The analysis includes:
1. Cleaning and preprocessing the data.
2. Extracting date and time features.
3. Analyzing sales trends by month.
4. Calculating total sales by month.
5. Determining the quantity ordered by product.
6. Cleaning and analyzing address data.
7. Visualizing sales data for various dimensions.

## Setup Instructions

### Prerequisites
- Python 3.x
- PySpark
- findspark
- Matplotlib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sales-data-analysis.git
    cd sales-data-analysis
    ```
2. Install the required Python packages:
    ```bash
    pip install pyspark findspark matplotlib
    ```

## Data Cleaning and Preparation
The data cleaning and preparation steps include:
- Dropping rows with missing values.
- Converting the `Order Date` column to a proper datetime format.
- Extracting additional time-related features such as hour, minute, full weekday name, and month.
- Cleaning the `Purchase Address` column to extract city and state information.

## Data Analysis
The analysis includes:
- Grouping data by month and calculating the total quantity ordered.
- Calculating the total sales by month.
- Grouping data by product to determine the total quantity ordered for each product.
- Grouping data by city to calculate total sales per city.
- Analyzing sales trends by hour of the day and day of the week.
- Identifying orders with multiple items and analyzing them.

## Visualization
Visualizations are created using Matplotlib to show:
1. Monthly quantity ordered.
2. Monthly sales in USD.
3. Quantity ordered by product.
4. Sales per city.
5. Sales trends by hour.
6. Sales trends by weekday.
7. Total number of items ordered for each product by day of the week.

## Code
```python
# You may need to install below packages.
!pip install pyspark findspark

# Import necessary packages
import findspark
findspark.init()
import pyspark
import pyspark.sql.types as ptypes
import pyspark.sql.functions as funcs
import matplotlib.pyplot as plt

# Initialize Spark Session
ss = pyspark.sql.SparkSession.builder.master('local[*]').appName('Sales').getOrCreate()

# Read sales data
sales_path = 'SalesAnalysis.csv'
sales_data = ss.read.csv(sales_path, inferSchema=True, header=True)

# Data Cleaning and Preparation
sales_data_dropna = sales_data.dropna()
sales_data_dropna = (sales_data_dropna
                     .withColumn('Order Date', funcs.from_unixtime(funcs.unix_timestamp('Order Date', 'MM/dd/yy HH:mm')))
                     .withColumn('hour', funcs.date_format(sales_data_dropna['Order Date'], 'HH').cast(ptypes.IntegerType()))
                     .withColumn('minute', funcs.date_format(sales_data_dropna['Order Date'], 'mm').cast(ptypes.IntegerType()))
                     .withColumn('full_weekday', funcs.date_format(sales_data_dropna['Order Date'], 'EEEE'))
                     .withColumn('month_ordered', funcs.date_format(sales_data_dropna['Order Date'], 'MMM')))

# Grouping by month and calculating quantity ordered
monthly_quantity = (sales_data_dropna.groupby('month_ordered')
                    .agg(funcs.count(funcs.col('Quantity Ordered')).alias('Quantity Ordered'))
                    .toPandas())
monthly_quantity.plot(kind='bar', x='month_ordered', xlabel='Month', ylabel='Quantity')

# Adding total_pay column
sales_data_dropna = sales_data_dropna.withColumn('total_pay', (funcs.col('Price Each') * funcs.col('Quantity Ordered')))

# Grouping by month and calculating total sales
monthly_highest_sales = (sales_data_dropna.groupby('month_ordered')
                         .sum('total_pay')
                         .withColumnRenamed('sum(total_pay)', 'monthly_total')
                         .toPandas())
monthly_highest_sales.plot(kind='bar', x='month_ordered', xlabel='Month', ylabel='Sales in USD ($)')

# Grouping by product and calculating quantity ordered
product_quantity = (sales_data_dropna.groupby('Product')
                    .agg({'Quantity Ordered': 'sum'})
                    .toPandas())
product_quantity.plot(kind='bar', x='Product', xlabel='Products', ylabel='Quantity Ordered')

# Cleaning address data
def clean_addresss(address):
    city = address.split(',')[1].strip()
    state = address.split(',')[2].split(' ')[1].strip()
    return f'{city} ({state})'

clean_addresss_udf = funcs.udf(lambda addrs: clean_addresss(addrs))
sales_data_dropna = sales_data_dropna.withColumn('city', clean_addresss_udf('Purchase Address'))

# Grouping by city and calculating total sales
city_sales = (sales_data_dropna.groupBy('city')
              .sum('total_pay')
              .withColumnRenamed('sum(total_pay)', 'total_sales_per_city')
              .toPandas())
city_sales.plot(kind='bar', x='city', xlabel='City', ylabel='Sales in USD ($)')

# Analyzing sales trends by hour
hourly_sales = (sales_data_dropna.groupby('hour')
                .sum('total_pay')
                .withColumnRenamed('sum(total_pay)', 'total_sales_per_hour')
                .sort('hour')
                .toPandas())
hourly_sales.plot(x='hour', xlabel='Hour', ylabel='Sales in USD ($)', grid=True)

# Analyzing sales trends by weekday
weekday_sales = (sales_data_dropna.groupby('full_weekday')
                 .sum('total_pay')
                 .withColumnRenamed('sum(total_pay)', 'total_sales_in_weekdays')
                 .toPandas())
weekday_sales.plot(x='full_weekday', xlabel='Week Days', ylabel='Sales in USD ($)', grid=True)

# Analyzing quantity ordered by product on Mondays
result_df = (sales_data_dropna.groupby('Product')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_orderd_in_year'))
             .where(funcs.col('full_weekday') == 'Monday')
             .select('Product', 'total_no_of_items_orderd_in_year')
             .sort('total_no_of_items_orderd_in_year', ascending=False))
result_pd_df = result_df.toPandas()
plt.figure(figsize=(10, 6))
plt.bar(result_pd_df['Product'], result_pd_df['total_no_of_items_orderd_in_year'])
plt.xlabel('Product')
plt.ylabel('Total Number of Items Ordered')
plt.title('Total Number of Items Ordered for Each Product on Mondays')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyzing quantity ordered by product for each weekday
result_df = (sales_data_dropna.groupby('Product', 'full_weekday')
             .agg(funcs.sum(funcs.col('Quantity Ordered')).alias('total_no_of_items_ordered'))
             .sort('Product', 'full_weekday'))
result_pd_df = result_df.toPandas()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(12, 6))
for weekday in weekdays:
    day_data = result_pd_df[result_pd_df['full_weekday'] == weekday]
    plt.plot(day_data['Product'], day_data['total_no_of_items_ordered'], marker='o', label=weekday)
plt.xlabel('Product')
plt.ylabel('Total Number of Items Ordered')
plt.title('Total Number of Items Ordered for Each Product by Day of the Week')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
