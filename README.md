# Beer Classification Project 
#### By Daniel Northcutt

<hr style="border:1px solid black"> </hr>

## Motivation: 

The main question this classification project aimed to ask is: are beer styles actually representative of shared traits of a particular style or is it that style boundaries more or less arbitrary?  For example, is the style stout actually distinguishable from other beers or is it just a stout because the label says so?  Bringing domain knowledge of 10 years as a brewer I came in to this project with a deep resepct for styles and tradition but also knowing many breweries will call their beer what they like.  With knowledge applied, I put this dataset to the test to see if classification ML models can predict a beers style


## Project Goal:
The goal of this project is to use machine learning classification models to test if they can accurately predict a beers style by specific attributes of the beer itself.  


## Project Description
This project pulls from 603 JSON files (beer-database-json) giving a database of 30,140 with various attributes.  The dataset was quite messy requiring lots of imputing and cleaning of values using python as the coding language. Most information was imputed from the 'style' column aka the beer's label description.  SRM (color from 1-40), IBU(international bitterness 1-120), and ABV (alcohol by volume) were the main descriptive features represented as floats.  Following specific attributes were pulled into boolean values (Trues/Falses) for characteristics such as contains wheat, smoke flavor, piney flavor, barrel aged, etc.  These features combined allowed for exploratory analysis with statistical testing for validation and eventual modeling.  Baseline was set at 73.6% representing the sum of the top 14 style classifications.  Models successsfuly beat baseline thus providing the value of a beer's particular attributes for a specific style.

The dataset imputed Nan values, removed outliers, scaled features, and split the data for proper analysis and modeling technqiues.
Note, models were tested on the grouping of 25 style classifications as well as 49 target variables without overfitting using Random Forest modeling. 

<hr style="border:1px solid black"> </hr>

## Intial Questions:
  - Does beer style represent particular attributes or is it an arbitrary description
  - How does SRM (color) define a style
  - How does ABV define a style
  - How does IBU (bitterness) define a style


## Data Dictionary

# Data Dictionary
| Feature                    | Datatype               | Description                                                           |
|:---------------------------|:-----------------------|:----------------------------------------------------------------------|
ID                           |          object  | Row name
Name                         |          object        | beer name
calculatedfinishedsquarefeet |          float64
fips                         |          float64
latitude                     |          float64
longitude                    |          float64
lotsizesquarefeet            |          float64
regionidcity                 |          float64
regionidcounty               |           float64
regionidzip                  |           float64
yearbuilt                    |           float64
structuretaxvaluedollarcnt   |           float64
taxvaluedollarcnt            |           float64
landtaxvaluedollarcnt        |           float64
taxamount                    |           float64
county                       |            object
age                          |           float64
age_bin                      |           float64
taxrate                      |           float64
acres                        |           float64
acres_bin                    |           float64
sqft_bin                     |           float64
structure_dollar_per_sqft    |           float64
structure_dollar_sqft_bin    |           float64
land_dollar_per_sqft         |           float64
lot_dollar_sqft_bin          |           float64
bath_bed_ratio               |           float64
cola                         |             int64
logerror_bins                |          category
baseline                     |           float64
scaled_latitude              |           float64
scaled_longitude             |           float64
scaled_bathroomcnt           |           float64
scaled_taxrate               |           float64
scaled_bedroomcnt            |           float64
scaled_lotsizesquarefeet     |           float64
scaled_age                   |           float64
scaled_acres                 |           float64
scaled_bath_bed_ratio        |           float64
scaled_calculatedfinishedsquarefeet|     float64
area_cluster                 |            object
size_cluster                 |            object
price_cluster                |            object
tax_cluster                  |            object
area_cluster_la_newer        |             uint8
area_cluster_la_older        |             uint8
area_cluster_northwest_costal|             uint8
area_cluster_palmdale_landcaster |         uint8
area_cluster_santa_clarita   |             uint8
area_cluster_se_coast        |             uint8
size_cluster_1250_to_1650    |             uint8
size_cluster_1300_to_2000    |             uint8
size_cluster_1500_to_1900    |             uint8
size_cluster_1500_to_2800    |             uint8
size_cluster_2300_to_4400    |             uint8
size_cluster_2900_to_4000    |             uint8
size_cluster_900_to_1200     |             uint8
price_cluster_144000_to_355000|            uint8
price_cluster_34000_to_110000 |            uint8
price_cluster_420000_to_870000|            uint8
price_cluster_45000_to_173000 |            uint8
price_cluster_69000_to_210000 |            uint8
tax_cluster_1000_to_3000      |            uint8
tax_cluster_16000_to_22000    |            uint8
tax_cluster_30000_to_40000    |            uint8
tax_cluster_5000_to_6000      |            uint8
tax_cluster_8500_to_12000     |            uint8
logerror                      |          float64




## Steps to Reproduce 
