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
restaurant_info = pd.read_csv("restaurantinfo.csv")

# load speech act categorization network
model = keras.models.load_model(model.h5) 
wordDict = 
catDict = 
catDictReverseLookup = {v: k for k, v in catDict.items()}

# initialize variables
speech_act = ""
last_said = "Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?"
last_suggested = 0

data = {'preferences':["","",""], 'order': [-1,-1,-1]}
preference = pd.DataFrame(data=data, index = ['food', 'area', 'price'])
food_types = pd.read_csv('food_type.csv')
price_types = pd.read_csv('price_type.csv')
location_types = pd.read_csv('location_type.csv')


# start conversation
def conversation():
    user_input = input('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like? \n')
    
    while speech_act != "bye" and speech_act != "thankyou":
        # identify speech act
        user_input = re.sub(r'[^\w\s]', '', user_input)
        speech_act = find_speechact(user_input.lower())
        
        # respond to speech act
        user_input = respond_to_user(speech_act)
        
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
    #identify preferences
    food_preference = list(set(inputText) & set(np.concatenate(food_types.values.tolist(), axis=0)))
    price_preference = list(set(inputText) & set(np.concatenate(price_types.values.tolist(), axis=0)))
    location_preference = list(set(inputText) & set(np.concatenate(location_types.values.tolist(), axis=0)))

    if food_preference != []:
        preference.set_value('food', 'preferences', food_preference[0])
        preference.set_value('food', 'order', max(preference['order']+1))
    if price_preference != []:
         preference.set_value('price', 'preferences', price_preference[0])
         preference.set_value('price', 'order', max(preference['order']+1))
    if location_preference != []:
         preference.set_value('area', 'preferences', location_preference[0])
         preference.set_value('area', 'order', max(preference['order']+1))
       
    # find set of restaurants that are in line with the preferences    
    unknown = ['area','price','food']
    pref_info = restaurant_info
    if preference.get_value('area', 'preferences') is not "":
        pref_info = pref_info[pref_info['area']==preference.get_value('area', 'preferences')]
        unknown.remove('area')
    if preference.get_value('food', 'preferences') is not "":
        pref_info = pref_info[pref_info['food']==preference.get_value('food', 'preferences')]
        unknown.remove('food')
    if preference.get_value('price', 'preferences') is not "":
        pref_info = pref_info[pref_info['pricerange']==preference.get_value('price', 'preferences')]
        unknown.remove('price')
    
    #give result
    #TODO: hou rekening met order en met wat allemaal gespecificeerd is
    if (len(restaurants) <= 10 and len(restaurants)>0) or unknown = []:
        last_said = (restaurants[last_suggested] + ' is an ' + price + ' restaurant serving ' + food + ' food in the ' + area + ' part of town')
        return(input(last_said + '\n'))
    
    #if there are to many options and not all of the criteria are provided
    else if len(restaurants) > 10:
        criteria = random.choice(unknown)
        last_said=('What kind of ' + criteria + ' would you like?')
        return(input(last_said + '\n'))

    #if there are no options in the data base
    else if len(restaurants) == 0:
        last_said =('Sorry i could not find a restaurant with your preferences, do you have something else you would like?')
        return(input(last_said + '\n'))
    

def request():
    #TODO: request = input('user: '), we should find a way to recognize the type of request of the user directly. 
    #TODO: wat als gevraagde info onbekend is? 
    if request == 'phone number':
        last_said = ('The phone number is ' + phones[last_suggested])
        return(input(last_said + '\n'))
    
    if request == 'adress':
        last_said =('The adress is ' + adresses[last_suggested] + ' '  + postcodes[last_suggested])
        return(input(last_said + '\n'))
    
def thankyou():
    print("You're welcome. Goodbye!")
    exit()
    
def null():
    last_said = "Sorry, I don't understand what you mean"
    return(input(last_said + '\n'))
    
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
    return(input(last_said + '\n'))
    
    
def hello():
    last_said = "Hello, what type of food, price range and area are you looking for?"
    return(input(last_said + '\n'))
    
    
def negate():
    
    
def repeat():
    return(input(last_said + '\n'))
    
def reqmore():
    print("%s is a %s priced restaurant serving %s food in the %s part of town" % (restaurants[last_suggested], price, food, area))
    last_said = ('Its phone number is ' + phones[last_suggested] + " and its adress is " + adresses[last_suggested] + ' '  + postcodes[last_suggested])
    return(input(last_said + '\n'))
    
def restart():
    last_said = ('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')
    return(input(last_said + '\n'))

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
conversation()
