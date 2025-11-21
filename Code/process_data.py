
import pandas as pd

data = pd.read_csv("E:\\Masters\\ML\\Assignment\\wallacecommunications.csv")
data = data[data['last_contact_this_campaign_month'] != 'j']
# drop ID and Country column as ID is just identifier and country is now all UK
data = data.drop(columns= ['ID'])
# Replace the incorrect values to correct value
data['has_tv_package'] = data['has_tv_package'].replace({'n':'no'})
data['last_contact'] = data['last_contact'].replace({'cell':'cellular'})

numerical_columns = ['age', 'current_balance', 'last_contact_this_campaign_day', 
                     'this_campaign', 'days_since_last_contact_previous_campaign', 'contacted_during_previous_campaign']

categorical_columns = ['country', 'married', 'job','conn_tr', 'education', 'town', 'last_contact',
                       'last_contact_this_campaign_month', 'outcome_previous_campaign','arrears',
                       'housing', 'has_tv_package']
