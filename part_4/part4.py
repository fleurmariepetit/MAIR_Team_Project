#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:48:52 2018

@author: gevelingbm
"""

import pandas as pd


# read in restaurant data
restaurantinfo = pd.read_csv('restaurantinfo.csv')

# keep track of the user responses
# TODO: Aan Kevin vragen of hij dat al had gedaan

# 
welcome = "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area , price range or food type . How may I help you?"
noOptions = "I'm sorry but there is no restaurant " # hier user preferences invullen