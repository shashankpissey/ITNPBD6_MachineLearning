
import pandas as pd

data = pd.read_csv("data/wallacecommunications.csv")
# drop the month value with 'j' as it can be January, June, July
data = data[data['last_contact_this_campaign_month'] != 'j']
# drop ID as it is just identifier
data = data.drop(columns= ['ID'])
# Replace the incorrect values to correct value
data['has_tv_package'] = data['has_tv_package'].replace({'n':'no'})
data['last_contact'] = data['last_contact'].replace({'cell':'cellular'})

numerical_columns = ['age', 'current_balance', 'last_contact_this_campaign_day', 
                     'this_campaign', 'days_since_last_contact_previous_campaign', 'contacted_during_previous_campaign']

categorical_columns = ['country', 'married', 'job','conn_tr', 'education', 'town', 'last_contact',
                       'last_contact_this_campaign_month', 'outcome_previous_campaign','arrears',
                       'housing', 'has_tv_package']
