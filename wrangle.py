import pandas as pd
import os
from sklearn.model_selection import train_test_split


###################################################
################## ACQUIRE DATA ###################
###################################################

def new_click_through_data():
    col_names = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 
             'site_domain', 'site_category', 'app_id', 'app_domain', 
             'app_category', 'device_id', 'device_ip', 'device_model',
             'device_type', 'device_conn_type', 'C14','C15','C16','C17','C18','C19','C20','C21']

    df = pd.read_csv('https://raw.githubusercontent.com/interviewquery/takehomes/oreilly_1/oreilly_1/training.csv', names=col_names)

    return df

def acquire_click_through_data():
    ''' 
    Checks to see if there is a local copy of the data, 
    if not go get data from github
    '''
    
    filename = 'training.csv'
    
    #if we don't have cached data or we want to get new data go get it from server
    if (os.path.isfile(filename) == False):
        df = new_click_through_data()
        #save as csv
        df.to_csv(filename,index=False)

    #else used cached data
    else:
        df = pd.read_csv(filename)

    return df

def prep_data(df):
    df['hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
    df['hour_of_day'] = df['hour'].dt.strftime('%H')
    df['day_of_week'] = df['hour'].dt.strftime('%A')
    df['hour_of_day'] =  df['hour_of_day'].astype(int)
    df['working_hours'] = (df['hour_of_day'] >= 8) & (df['hour_of_day'] < 18)


    return df

def split_data(df, test_size=0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(df, stratify=df['click'], test_size = test_size , random_state=27)
    train, validate = train_test_split(train,  stratify=train['click'], test_size = (test_size + 0.05)/(1-test_size), random_state=27)
    
    return train, validate, test