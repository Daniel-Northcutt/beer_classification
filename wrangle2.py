##### THIS IS THE SAME AS WRANGLE 1 BUT DROPPING TARGET VARIABLE NUMBER

import seaborn as sns
import json
import os
import codecs
import csv
import re
import os
import warnings
import pandas as pd
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

    # changing to float - don't need if I place df['style'].astype(float)
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
    
    df['style'] = df['style'].astype(str)
    
    # This only means it has the keyword in the description
    df['sour'] = df['description'].str.contains('sour')
    df['sour'] = df['sour'].astype(bool)

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
    # American hops used
    df['american_hop'] = df['style'].str.contains('American-variety hop character')

    # piney flavor
    df['piney_flavor'] = df['style'].str.contains('piney')

    # Belgian characteristics
    df['belgian'] = df['style'].str.contains('Belgian')

    # Is Imperial
    df['imperial'] = df['style'].str.contains('imperial')

    # Honey flavor
    df['honey'] = df['style'].str.contains('honey')

    #Ester flavor (yeasty)
    df['esters'] = df['style'].str.contains('esters')

    #bitterness in description
    df['bitterness'] = df['style'].str.contains('bitterness')

    #Oak characteristics
    df['oak'] = df['style'].str.contains('oak')


    ## Will add these if I don't dropna on srm and ibu values
    # #has energy added
    # df['has_energy'] = df['description'].str.contains('energy')

    # #has brett yeast
    # df['has_brett'] = df['description'].str.contains('brett')
    
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

    ###### Look to add wood, hoppy, (think of other keywords)



    # #### I would like to include the NaNs for ibu and srm but will look at it after the MVP
    # #new column
    df['style_collapsed'] = df['short_name']
    ### GROUPING SIMILAR STYLES INTO BROADER CATEGORIES
    df['style_collapsed'] = df['style_collapsed'].replace(['American IPA','Juicy or Hazy IPA', 'Session IPA'], 'IPA')
    df['style_collapsed'] = df['style_collapsed'].replace(['Juicy or Hazy Double IPA','Imperial IPA'], 'Imperial_IPA')
    df['style_collapsed'] = df['style_collapsed'].replace(['American Barleywine', 'British Barleywine'], 'Barleywine')
    df['style_collapsed'] = df['style_collapsed'].replace(['Wheat Ale','Fruit Wheat Ale',  'Wheatwine',  'Hefeweizen', 'Dark Wheat Ale', 'Dunkelweizen','Kristallweizen', 'Weizenbock','Leichtesweizen', 'Bernsteinfarbenesweizen'], 'Wheat_Beers')
    df['style_collapsed'] = df['style_collapsed'].replace(['American Lager', 'American Premium Lager', 'American Light Lager','Vienna Lager','American Low-Carb Lager', 'Tropical Light Lager','American Ice Lager','Dry Lager','Kellerbier'], 'Lager')
    df['style_collapsed'] = df['style_collapsed'].replace(['Blonde','Belgian Blonde'], 'Blonde')

    df['style_collapsed'] = df['style_collapsed'].replace(['American Imperial Stout', 'American Stout', 'Sweet Stout', 'Oatmeal Stout', 'Stout','Dry Irish Stout', 'British Imperial Stout', 'Export Stout'], 'Stout')


    df['style_collapsed'] = df['style_collapsed'].replace([ 'Brown Porter', 'Robust Porter','American Imperial Porter', 'Baltic Porter', 'Smoke Porter'], 'Porter')


    df['style_collapsed'] = df['style_collapsed'].replace(['American Pale', 'English Pale', 'English Pale Mild', 'American Strong Pale', 'International Pale', 'Austrailian Pale', 'Juicy or Hazy Pale Ale', 'Wet Hop Ale','American/Belgian Pale', 'Classic Australian Pale'], 'Paleale')


    df['style_collapsed'] = df['style_collapsed'].replace([ 'Amber','American Amber Lager', 'Leichtbier'], 'Amber')

    df['style_collapsed'] = df['style_collapsed'].replace([ 'Belgian Pale', 'Belgian Pale Strong', 'Belgian Ale', 'Bière de Garde',  'Belgian Dubbel', 'Belgian Dark Strong', 'Belgian Tripel', 'Belgian Fruit' ,'Belgian Quad', 'Belgian Table Beer'], 'Belgian_Variety')


    df['style_collapsed'] = df['style_collapsed'].replace(['International Pilsener','Bohemian Pilsener','Contemporary American Pilsener', 'German Pilsener','American Pilsener', 'Helles', 'Dortmunder']
    , 'Pilsner')

    df['style_collapsed'] = df['style_collapsed'].replace([ 'Sour', 'Lambic', 'Gueuze','Leipzig Gose', 'Fruit Lambic', 'Berlinerweisse','Contemporary Gose']
    , 'Sour_Gose')

    df['style_collapsed'] = df['style_collapsed'].replace(['Double Red', 'Imperial Red','Irish Red','Flanders Red']
    , 'Redale')

    df['style_collapsed'] = df['style_collapsed'].replace(['BBL Aged Dark', 'BBL Aged Pale', 'BBL Aged Sour', 'BBL Aged','BBL Aged Strong']
    , 'BBL_aged')

    df['style_collapsed'] = df['style_collapsed'].replace([ 'English Brown', 'American Brown'], 'Brown')

    df['style_collapsed'] = df['style_collapsed'].replace(['Rauchbier','Märzen Rauchbier', 'Helles Rauchbier','Bock Rauchbier', 'Smoke Beer'], 'Smoke')

    df['style_collapsed'] = df['style_collapsed'].replace([ 'Schwarzbier', 'American Dark Lager','Euro Dark','English Dark Mild','American/Belgian Dark', 'Black Ale'], 'Darkale')


    df['style_collapsed'] = df['style_collapsed'].replace(['Chili Beer','Coffee Beer','Spice Beer', 'Chocolate Beer', 'Pumpkin Beer','Honey Beer'], 'Spice_Flavor_Ales')


    df['style_collapsed'] = df['style_collapsed'].replace([ 'Scottish Export','Scottish Heavy','Old Ale', 'Scotch Ale','Strong Ale'], 'Strong_ales')
    df['style_collapsed'] = df['style_collapsed'].replace([ 'Grodziskie', 'Session'], 'Mild_ales')
    df['style_collapsed'] = df['style_collapsed'].replace(['ESB', 'Special Bitter','Bitter'], 'English_Bitter')
    df['style_collapsed'] = df['style_collapsed'].replace([ 'American Oktoberfest','Oktoberfest'], 'Oktoberfest')
    df['style_collapsed'] = df['style_collapsed'].replace(['Rye Ale','German Rye'], 'Rye_Ales')
    df['style_collapsed'] = df['style_collapsed'].replace(['Scottish Light', 'English Summer Ale'], 'English_Mild')
    
    # Remove any values under 20 as it will be to difficult to classify
    style_counts = df['style_collapsed'].value_counts()
    to_remove_counts = style_counts[style_counts <=200].index
    df = df[~df.style_collapsed.isin(to_remove_counts)]
    
    
    return df