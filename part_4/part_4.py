import csv
import random
import pandas as pd
import keras
import numpy as np

"""
Print welcome
Recognize speech act of user input
respond according to speech act
"""

# read in restaurant data
restaurant_info = csv.reader(open("restaurantinfo.csv"), delimiter=",")

# load speech act categorization network
model = keras.models.load_model(model.h5) 
wordDict = 
catDict = 
catDictReverseLookup = {v: k for k, v in catDict.items()}

# initialize variables
speech_act = ""
last_said = "Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?"
last_suggested = 0

data = {'types':['food', 'area', 'price'], 'preferences':["","",""], 'order': [9,9,9]}
preference = pd.DataFrame(data=data)

restaurants = []
phones = []
adresses = []
postcodes = []

# start conversation
user_input = input('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like? \n')

while speech_act != "bye" and speech_act != "thankyou":
    # identify speech act
    #TODO: Find speechact: gebruik user input en getrainde netwerk van part 2
    speech_act = find_speechact(user_input)
    
    # respond to speech act
    respond_to_user(speech_act)
    
def find_speechact(user_input):
    userInputTransformed = transformUserSentence(wordDict, user_input)
    inputStructure = np.ndarray(shape=(1, 23))
    inputStructure[0] = userInputTransformed

    prediction = model.predict_classes(inputStructure)[0]
    return(catDictReverseLookup[prediction + 1])    
        
def transformUserSentence(wordDict, sentence):
    transformedSentence = np.ones((23,), dtype=int)
    wordCount = 0

    for word in sentence.split():
        if word in wordDict:
            transformedSentence[wordCount] = wordDict[word]
        else:
            transformedSentence[wordCount] = 0
        wordCount += 1
        if wordCount > 23:
            break

    return transformedSentence
    
def respond_to_user(speech_act):
    switcher = {
            "inform": inform,
            "request": request,
            "thankyou": thankyou,
            "null": null,
            "requalts": requalts,
            "affirm": affirm,
            "bye": bye,
            "ack": ack,
            "hello": hello,
            "negate": negate,
            "repeat": repeat,
            "reqmore": reqmore, 
            "restart": restart,
            "confirm": confirm,
            "deny": deny
            }

    func = switcher.get(speech_act, "speechact unknown")
    return func()

#TODO: Definieer elke speechact functie
def inform():
    # TODO: zorg dat het ook werkt als niet alle criteria gegeven zijn, en bij vragen om meer info moet hij rekening houden met wat hij al weet
    for row in restaurant_info:
    #if all of the criteria are provided
        if price == row[1] and area == row[2] and food == row[3]:
            print(row)
            restaurants.append(row[0])
            phones.append(row[4])
            adresses.append(row[5])
            postcodes.append(row[6])

    #give result
    if len(restaurants) <= 10 and len(restaurants)>0:
        # TODO: or all 3 things are specified
        print (restaurants[last_suggested] + ' is an ' + price + ' restaurant serving ' + food + ' food in the ' + area + ' part of town')
    
    #if there are to many options and not all of the criteria are provided
    if len(restaurants) > 10:
        criteria = random.choice([' area',' price',' food'])
        print('What kind of ' + criteria + ' would you like?')

    #if there are no options in the data base
    if len(restaurants) == 0:
        print('Sorry i could not find a restaurant with your preferences, do you have something else you would like?')

def request():
    #TODO: request = input('user: '), we should find a way to recognize the type of request of the user directly. 
    if request == 'phone number':
        print('The phone number is ' + phones[last_suggested])
    if request == 'adress':
        print('The adress is ' + adresses[last_suggested] + ' '  + postcodes[last_suggested])

def thankyou():
    print("You're welcome. Goodbye!")
    exit()
    
def null():
    last_said = "Sorry, I don't understand what you mean"
    print(last_said)
    
def reqalts():
    last_suggested += 1
    if last_suggested < len(restaurants):
        give_suggestion(last_suggested)
    else:
        print('Sorry there are no other restaurants with your preferences. I will repeat the earlier suggestions.')
        last_suggested = 0
        give_suggestion(last_suggested)
        
def affirm():
    
    
def bye():
    exit()
    
def ack():
    last_said = "Is there anything else you want to know?"
    print(last_said)
    
def hello():
    last_said = "Hello, what type of food, price range and area are you looking for?"
    print(last_said)
    
def negate():
    
    
def repeat():
    print(last_said)
    
def reqmore():
    print("%s is a %s priced restaurant serving %s food in the %s part of town" % (restaurants[last_suggested], price, food, area))
    print('Its phone number is ' + phones[last_suggested] + " and its adress is " + adresses[last_suggested] + ' '  + postcodes[last_suggested])
    
def restart():
    print('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')

def confirm():
    
    
def deny():
    
    
def give_suggestion(last_suggested):
    # TODO: suggesties moeten prijs, food en area in volgorde van opgeven geven.
    print (restaurants[last_suggested] + ' is an ' + price + ' restaurant serving ' + food + ' food in the ' + area + ' part of town')
    
      
    
"""
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



"""

