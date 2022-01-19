import pandas as pd
import numpy as np
import seaborn as sns
import json
import os
import codecs
import csv
import re
import os
import warnings
warnings.filterwarnings('ignore')

################################
#creating an empty dataframe, looping through files, appending to dataframe - all jsons pulled into one
def acquire_beer_classification():
    df1 = pd.DataFrame()
    for file in os.listdir("/Users/daniels/codeup-data-science/beer_classification/beer-database"):
        if file.endswith(".json"):
            with open ("/Users/daniels/codeup-data-science/beer_classification/beer-database/" + str(file)) as f:            
                data = json.load(f)
                df = pd.DataFrame(data['data'])
                df1 = df.append(df1)
                #df = df1
    return df1
    #### 30140 rows & 28 columns

################################
def prepare_beer(df):
    #drop columns
    columns_drop = ['createDate', 'updateDate', 'available','servingTemperature', 'servingTemperatureDisplay', 'foodPairings',
    'year', 'availableId', 'labels', 'status', 'isRetired', 'isOrganic', 'glass', 'beerVariationId', 'beerVariation', 'glasswareId','srmId','originalGravity']
    df = df.drop(columns_drop, 1)
    
    # pulling values from style string
    df['ibu_min'] = df['style'].str.get('ibuMin')
    df['ibu_max'] = df['style'].str.get('ibuMax')
    df['abv_min'] = df['style'].str.get('abvMin')
    df['abv_max'] = df['style'].str.get('abvMax')
    df['srm_min'] = df['style'].str.get('srmMin')
    df['srm_max'] = df['style'].str.get('srmMax')
    df['og_min'] = df['style'].str.get('ogMin')
    df['fg_min'] = df['style'].str.get('fgMin')
    df['fg_max'] = df['style'].str.get('fgMax')
    df['category_id'] = df['style'].str.get('categoryId')
    df['short_name'] = df['style'].str.get('shortName')

    # changing to float
    df['ibu_min'] = df['ibu_min'].astype(float)
    df['ibu_max'] = df['ibu_max'].astype(float)
    df['abv_min'] = df['abv_min'].astype(float)
    df['abv_max'] = df['abv_max'].astype(float)
    df['srm_min'] = df['srm_min'].astype(float)
    df['srm_max'] = df['srm_max'].astype(float)
    df['og_min'] = df['og_min'].astype(float)
    df['fg_min'] = df['fg_min'].astype(float)
    df['fg_max'] = df['fg_max'].astype(float)
    df['category_id'] = df['category_id'].astype(int)

    # creating avg based on min and max values given
    df['srm_avg'] = (df['srm_min'] + df['srm_max'])/2
    df['ibu_avg'] = (df['ibu_min'] + df['ibu_max'])/2
    df['abv_avg'] = (df['abv_min'] + df['abv_max'])/2

    # pulling values of srm from string - change to float
    df['srm'] = df['srm'].str.get('id')
    df['srm'] = df['srm'].astype(float)
    
    
    # This only means it has the keyword in the description

    df['sour_des'] = df['description'].str.contains('sour')
    df['sour_des'] = df['sour_des'].astype(bool)

    # has fruit description
    df['fruit_des'] = df['description'].str.contains('fruit')
    df['fruit_des'] = df['fruit_des'].astype(bool)

    # has wheat description
    df['wheat_des'] = df['description'].str.contains('wheat')
    df['wheat_des'] = df['wheat_des'].astype(bool)

    # has smoke description
    df['smoke_des'] = df['description'].str.contains('smoke')
    df['smoke_des'] = df['smoke_des'].astype(bool)

    # has smoke description
    df['chili_des'] = df['description'].str.contains('chili')
    df['chili_des'] = df['chili_des'].astype(bool)

    # has mead description
    df['mead_des'] = df['description'].str.contains('mead')
    df['mead_des'] = df['mead_des'].astype(bool)

    #has lager ferment
    df['lager_des'] = df['description'].str.contains('lager')
    df['lager_des'] = df['lager_des'].astype(bool)

    # is barrel aged
    df['BBL'] = df['description'].str.contains('BBL')
    df['BBL'] = df['BBL'].astype(bool)

    #has energy added
    df['has_energy'] = df['description'].str.contains('energy')
    df['has_energy'] = df['has_energy'].astype(bool)

    #has brett yeast
    df['has_brett'] = df['description'].str.contains('brett')
    df['has_brett'] = df['has_brett'].astype(bool)
    
    
    
    #merge 
    df['srm'] = df['srm'].fillna(df['srm_avg'])
    df['ibu'] = df['ibu'].fillna(df['ibu_avg'])
    df['abv'] = df['abv'].fillna(df['abv_avg'])
    

    # dropping abv null values
    # now to 29929
    df = df[pd.notnull(df['abv'])]

    # 3000 null values
    df = df[pd.notnull(df['srm'])]

    # 768 null values
    df = df[pd.notnull(df['ibu'])]
    
    #### I would like to include the NaNs for ibu and srm but will look at it after the MVP
    return df


    ###### Look to add wood, hoppy, (think of other keywords)