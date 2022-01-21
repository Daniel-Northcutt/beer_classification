# Beer Classification Project 
#### By Daniel Northcutt

<hr style="border:1px solid black"> </hr>

## Motivation: 

The main question this classification project aimed to ask is: are beer styles actually representative of shared traits of a particular style or is it that style boundaries more or less arbitrary?  For example, is the style stout actually distinguishable from other beers or is it just a stout because the label says so?  Bringing domain knowledge of 10 years as a brewer I came in to this project with a deep resepct for styles and tradition but also knowing many breweries will call their beer what they like.  With knowledge applied, I put this dataset to the test to see if classification ML models can predict a beers style

## Executive Summary 
(to be completed)

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

## Project Planning
(to be completed)

## Steps to Reproduce 
(to be completed)


## Data Dictionary

# Data Dictionary
| Feature                    | Datatype               | Description                                                           |
|:---------------------------|:-----------------------|:----------------------------------------------------------------------|
id                            | object        | identification #
name                           | object       | name
nameDisplay              |        object      | name-displa
abv                         |    float64      | alcohol by volume
styleId                     |      int64      | style ID
statusDisplay                |    object      | verification for style ID
style                         |   object      | description 'beer's label'
description                  |    object      | 2ndary description (mostly Nan)
ibu                        |     float64      | International Bitterness Units
srm                       |      float64      | Beer's Color (0-40)
ibu_min                  |       float64      | ibu min for beer
ibu_max                  |       float64      | ibu max for beer
abv_min                   |      float64      | abv min for beer
abv_max                   |      float64      | abv max for beer
srm_min                   |      float64      | srm min for beer
srm_max                   |      float64      | srm max for beer
og_min                    |      float64      | og min for beer (starting gravity)
fg_min                   |       float64      | fg min for beer (finishing gravity)
fg_max                    |      float64      | fg max for beer (finishing gravity)
category_id                 |      int64      | category id
short_name                   |    object      | beer identification name
srm_avg                     |    float64      | srm average for beer
ibu_avg                    |     float64      | ibu average for beer
abv_avg                     |    float64      | abv average for beer
sour                        |       bool      | sour characteristics 
fruit_des                    |      bool      | fruit characteristics
wheat_des                    |      bool      | wheat characteristics
smoke_des                   |       bool      | smoke characteristics
chili_des                       |   bool      | chili charcteristics
mead_des                      |     bool      | mead characteristics
lager_des                     |     bool      | lager yeast characteristics
BBL                            |    bool      | barrel aged
american_hop                 |      bool      | american hop characteristics
piney_flavor                  |     bool      | piney flavor characteristics
belgian                       |     bool      | Belgian characteristics
imperial                     |      bool      | Imperial (higher abv)
honey                        |      bool      | honey characteristics
esters                      |       bool      | ester (yeasty) characteristics
bitterness                   |      bool      | bitterness attribute (typically for higher ibu)
oak                           |     bool      | oak characteristics
style_collapsed                 | object      | Beer Style Condensed (Target Variable) 

