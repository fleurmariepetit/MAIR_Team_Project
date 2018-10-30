import random
import pandas as pd
import keras
import numpy as np
import json
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

"""
Print welcome
Recognize speech act of user input
respond according to speech act
"""

# read in restaurant data
restaurant_info = pd.read_csv("restaurantinfo.csv")

# load speech act categorization network
model = keras.models.load_model('model7eps.h5') 
wordDict = json.loads(open('wordDict7.json').read())
catDict = json.loads(open('catDict7.json').read())
catDictReverseLookup = {v: k for k, v in catDict.items()}

# initialize variables
speech_act = ""
last_said = "Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?"
last_suggested = -1
user_input = ""

pref_info = pd.DataFrame()
data = {'preferences':["","",""], 'order': [-1,-1,-1]}
preference = pd.DataFrame(data=data, index = ['food', 'area', 'price'])
food_types = pd.read_csv('food_type.csv')
price_types = pd.read_csv('price_type.csv')
location_types = pd.read_csv('location_type.csv')

# start conversation
def conversation(speech_act):
    global user_input
    user_input = input('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like? \n')
    
    while speech_act != "bye" and speech_act != "thankyou":
        # identify speech act
        user_input = re.sub(r'[^\w\s]', '', user_input).lower()
        speech_act = find_speechact(user_input)
        
        # respond to speech act
        user_input = respond_to_user(speech_act)

        if user_input == "quit":
            break

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
            "reqalts": reqalts,
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

def inform():
    #identify preferences
    global pref_info
    global preference
    global last_said

    food_preference = list(set(user_input.split()) & set(np.concatenate(food_types.values.tolist(), axis=0)))
    price_preference = list(set(user_input.split()) & set(np.concatenate(price_types.values.tolist(), axis=0)))
    location_preference = list(set(user_input.split()) & set(np.concatenate(location_types.values.tolist(), axis=0)))

    if food_preference != []:
        preference.set_value('food', 'preferences', food_preference[0])
        preference.set_value('food', 'order', max(preference['order']+1))
    if price_preference != []:
         preference.set_value('price', 'preferences', price_preference[0])
         preference.set_value('price', 'order', max(preference['order']+1))
    if location_preference != []:
         preference.set_value('area', 'preferences', location_preference[0])
         preference.set_value('area', 'order', max(preference['order']+1))

    # TODO: It seems it doesn't remove from unknown. Keeps refreshing each loop.
    unknown = ['area', 'price', 'food']
    print(unknown)

    #If the user doesn't care
    if ('doesnt matter' or 'does not matter' or 'dont care' or 'do not care') in user_input:
        if 'area' in last_said:
            preference.set_value('area', 'preferences', 'irrelevant')
            unknown.remove('area')
            preference.set_value('area', 'order', max(preference['order'] + 1))
        elif 'price' in last_said:
            preference.set_value('price', 'preferences', 'irrelevant')
            unknown.remove('price')
            preference.set_value('price', 'order', max(preference['order'] + 1))
        elif 'food' in last_said:
            preference.set_value('food', 'preferences', 'irrelevant')
            unknown.remove('food')
            preference.set_value('food', 'order', max(preference['order'] + 1))

    # if anything is good
    if 'any' in user_input:
        list_of_words = user_input.split()
        next_word = list_of_words[list_of_words.index('any') + 1]
        if next_word == 'area':
            preference.set_value('area', 'preferences', 'irrelevant')
            unknown.remove('area')
            preference.set_value('area', 'order', max(preference['order'] + 1))
        if next_word == 'price':
            preference.set_value('price', 'preferences', 'irrelevant')
            unknown.remove('price')
            preference.set_value('price', 'order', max(preference['order'] + 1))
        if next_word == 'food':
            preference.set_value('food', 'preferences', 'irrelevant')
            unknown.remove('food')
            preference.set_value('food', 'order', max(preference['order'] + 1))

    # find set of restaurants that are in line with the preferences    
    pref_info = restaurant_info
    if preference.get_value('area', 'preferences') is not "":
        if preference.get_value('area', 'preferences') is not'irrelevant':
            pref_info = pref_info[pref_info['area']==preference.get_value('area', 'preferences')]
            unknown.remove('area')
    if preference.get_value('food', 'preferences') is not "":
        if preference.get_value('food', 'preferences') is not 'irrelevant':
            pref_info = pref_info[pref_info['food']==preference.get_value('food', 'preferences')]
            unknown.remove('food')
    if preference.get_value('price', 'preferences') is not "":
        if preference.get_value('price', 'preferences') is not 'irrelevant':
            pref_info = pref_info[pref_info['pricerange']==preference.get_value('price', 'preferences')]
            unknown.remove('price')
    pref_info['index'] = list(range(0,len(pref_info)))
    pref_info = pref_info.set_index('index')
    
    #suggest a restaurant
    if (len(pref_info) <= 10 and len(pref_info)>0) or unknown == []:
        global last_suggested
        last_suggested += 1
        return give_suggestion(last_suggested)
        
    #if there are to many options and not all of the criteria are provided
    elif len(pref_info) > 10:
        criteria = random.choice(unknown)
        last_said=('What kind of ' + criteria + ' would you like?')
        return(input(last_said + '\n'))

    #if there are no options in the data base
    elif len(pref_info) == 0:
        last_said =('Sorry i could not find a restaurant with your preferences, do you have something else you would like?')
        return(input(last_said + '\n'))

def request():
    global last_said

    # answer user's question
    if 'price' in user_input:
        last_said = ('The pricerange of the restaurant is ' + pref_info.get_value(last_suggested, 'pricerange'))
        return(input(last_said + '\n'))
    
    if 'area' in user_input:
        last_said = ('The restaurant is in the' + pref_info.get_value(last_suggested, 'area') + " of town.")
        return(input(last_said + '\n'))
    
    if 'food' in user_input:
        last_said = ('The restaurant serves ' + pref_info.get_value(last_suggested, 'food') + " food.")
        return(input(last_said + '\n'))
        
    if 'number' in user_input:
        if pd.isna(pref_info.get_value(last_suggested, 'phone'))== False:
            last_said =('The phone number is ' + pref_info.get_value(last_suggested, 'phone'))
            return(input(last_said + '\n'))
        else:
            last_said =('The phone number is unknown')
            return(input(last_said + '\n'))
    
    if 'address' in user_input:
        if pd.isna(pref_info.get_value(last_suggested, 'addr'))== False and pd.isna(pref_info.get_value(last_suggested, 'postcode'))== False:
            last_said =('The adress is ' + pref_info.get_value(last_suggested, 'addr') + ' '  + pref_info.get_value(last_suggested, 'postcode'))
            return(input(last_said + '\n'))
        elif pd.isna(pref_info.get_value(last_suggested, 'addr'))== False and pd.isna(pref_info.get_value(last_suggested, 'postcode'))== True:
            last_said =('The adress is ' + pref_info.get_value(last_suggested, 'addr'))
            return(input(last_said + '\n'))
        else:
            last_said =('The adress is unknown')
            return(input(last_said + '\n'))
            
    last_said = "Sorry, I don;t understand your request"
    return(input(last_said + '\n'))
    
def thankyou():
    print("You're welcome. Goodbye!")
    exit()
    
def null():
    global last_said

    last_said = "Sorry, I don't understand what you mean"
    return(input(last_said + '\n'))
    
def reqalts():
    global last_suggested

    last_suggested += 1
    if last_suggested < len(pref_info):
        return give_suggestion(last_suggested)
    else:
        print('Sorry there are no other restaurants with your preferences. I will repeat the earlier suggestions.')
        last_suggested = 0
        return give_suggestion(last_suggested)
        
def affirm():
    global last_suggested
    last_suggested += 1
    return give_suggestion(last_suggested)
    
def bye():
    exit()
    
def ack():
    global last_said

    last_said = "Is there anything else you want to know?"
    return(input(last_said + '\n'))
    
def hello():
    global last_said

    last_said = "Hello, what type of food, price range and area are you looking for?"
    return(input(last_said + '\n'))
    
def negate():
    global last_said

    last_said = "Is there anything else I can help you with?"
    return(input(last_said + '\n'))
    
def repeat():
    global last_said

    return(input(last_said + '\n'))
    
def reqmore():
    global last_said

    # TODO: Hou er rekening mee dat niet altijd alle info bekend is. 
    print("%s is a %s priced restaurant serving %s food in the %s part of town" % (pref_info.get_value(last_suggested, 'restaurantname'), pref_info.get_value(last_suggested, 'pricerange'), pref_info.get_value(last_suggested, 'food'), pref_info.get_value(last_suggested, 'area')))
    last_said = ('Its phone number is ' + pref_info.get_value(last_suggested, 'phone') + " and its adress is " + pref_info.get_value(last_suggested, 'addr') + ' '  + pref_info.get_value(last_suggested, 'postcode'))
    return(input(last_said + '\n'))
    
def restart():
    global last_said

    last_said = ('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')
    return(input(last_said + '\n'))

def confirm():
    global last_said

    conf_food = list(set(user_input.split()) & set(np.concatenate(food_types.values.tolist(), axis=0)))
    conf_price = list(set(user_input.split()) & set(np.concatenate(price_types.values.tolist(), axis=0)))
    conf_location = list(set(user_input.split()) & set(np.concatenate(location_types.values.tolist(), axis=0)))

    conf_t = True
    category = []

    if conf_food != []:
        if conf_food[0] != pref_info.get_value(last_suggested,'food'):
            conf_t = False
            category.append("food")

    if conf_price != []:
        if conf_price[0] != pref_info.get_value(last_suggested,'pricerange'):
            conf_t = False
            category.append("pricerange")

    if conf_location != []:
        if conf_location[0] != pref_info.get_value(last_suggested,'area'):
            conf_t = False
            category.append("area")

    cat_str = ""
    for e in category:
        e = str(e)
        if e == category[0]:
            cat_str = e
        else:
            cat_str = cat_str + ", " + e

    if conf_t:
        last_said = "Yes, I saved it correctly."
    else:
        last_said = "No, I am afraid you misunderstood the" + cat_str + "."

    return(input(last_said + '\n'))
    
def deny():
    global last_said

    last_said = "What are you exactly looking for? "
    return(input(last_said + '\n'))
    
def give_suggestion(last_suggested):
    global last_said

    first_sentence = ''
    second_sentence = ''
    third_sentence = ''

    food_sentence = ' serving ' + pref_info.get_value(last_suggested, 'food') + ' food'
    price_sentence = ' in the ' + pref_info.get_value(last_suggested, 'pricerange') + ' price range'
    area_sentence = ' in the ' + pref_info.get_value(last_suggested, 'area') + ' part of town'

    food_order = preference.get_value('food', 'order')
    if food_order == 0:
        first_sentence = food_sentence
    if food_order == 1:
        second_sentence = food_sentence
    if food_order == 2:
        third_sentence = food_sentence

    area_order = preference.get_value('area', 'order')
    if area_order == 0:
        first_sentence = area_sentence
    if area_order == 1:
        second_sentence = area_sentence
    if area_order == 2:
        third_sentence = area_sentence

    price_order = preference.get_value('price', 'order')
    if price_order == 0:
        first_sentence = price_sentence
    if price_order == 1:
        second_sentence = price_sentence
    if price_order == 2:
        third_sentence = price_sentence

    last_said = (pref_info.get_value(last_suggested,'restaurantname') + ' is a restaurant' + first_sentence + second_sentence + third_sentence + '.')
    return(input(last_said + '\n'))
    
conversation(speech_act)
