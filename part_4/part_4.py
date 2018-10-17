import csv
import random

print('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')
restaurant_info = csv.reader(open("restaurantinfo.csv"), delimiter=",")
food = input('What type of food would you like?')
price = input('What is your price range?')
area = input('What part of town?')
restaurants = []
phones = []
adresses = []
postcodes = []

for row in restaurant_info:
    #if all of the criteria are provided
    if price == row[1] and area == row[2] and food == row[3]:
        print(row)
        restaurants.append(row[0])
        phones.append(row[4])
        adresses.append(row[5])
        postcodes.append(row[6])

#give result
if len(restaurants) < 10:
    print (restaurants[0] + ' is an ' + price + ' restaurant serving ' + food + ' food in the ' + area + ' part of town')

#If the user requests info about the restaurant
request = input('user: ')
if request == 'phone number':
    print('The phone number is ' + phones[0])
if request == 'adress':
    print('The adress is ' + adresses[0] + ' '  + postcodes[0])

#if there are to many options and not all of the criteria are provided
if len(restaurants) > 10:
    criteria = random.choice([' area',' price',' food'])
    print('What kind of ' + criteria + ' would you like?')

#if there are no options in the data base
if len(restaurants) == 0:
    print('Sorry i could not find a restaurant with your preferences, do you have something else you would like?')




