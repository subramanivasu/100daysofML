#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://0.0.0.0:9696/Predict'

customer_id = 'ABC-29384'

customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}

response = requests.post(url, json=customer).json()
print(response)

if response['Churn'] == True:
    print('Sending promo email to %s' % customer_id)
else:
    print('Not sending promo email to %s' % customer_id)