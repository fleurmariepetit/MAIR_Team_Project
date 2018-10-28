import random
import pandas as pd
import keras
import numpy as np
import json
import re

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
    user_input = input('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like? \n')
    
    while speech_act != "bye" and speech_act != "thankyou":
        # identify speech act
        user_input = re.sub(r'[^\w\s]', '', user_input).lower()
        speech_act = find_speechact(user_input)
        
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
    last_said = "Is there anything else you want to know?"
    return(input(last_said + '\n'))
    
def hello():
    last_said = "Hello, what type of food, price range and area are you looking for?"
    return(input(last_said + '\n'))
    
def negate():
    last_said = "Is there anything else I can help you with?"
    return(input(last_said + '\n'))
    
def repeat():
    return(input(last_said + '\n'))
    
def reqmore():
    # TODO: Hou er rekening mee dat niet altijd alle info bekend is. 
    print("%s is a %s priced restaurant serving %s food in the %s part of town" % (pref_info.get_value(last_suggested, 'restaurantname'), pref_info.get_value(last_suggested, 'pricerange'), pref_info.get_value(last_suggested, 'food'), pref_info.get_value(last_suggested, 'area')))
    last_said = ('Its phone number is ' + pref_info.get_value(last_suggested, 'phone') + " and its adress is " + pref_info.get_value(last_suggested, 'addr') + ' '  + pref_info.get_value(last_suggested, 'postcode'))
    return(input(last_said + '\n'))
    
def restart():
    last_said = ('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')
    return(input(last_said + '\n'))

def confirm():
    conf_food = list(set(user_input.split()) & set(np.concatenate(food_types.values.tolist(), axis=0)))
    conf_price = list(set(user_input.split()) & set(np.concatenate(price_types.values.tolist(), axis=0)))
    conf_location = list(set(user_input.split()) & set(np.concatenate(location_types.values.tolist(), axis=0)))

    if conf_food != []:
        if conf_food[0] == pref_info.get_value(last_suggested,'food'):
            last_said = 'Yes, the restaurant servers %s food' %(pref_info.get_value(last_suggested,'food'))
            return(input(last_said + '\n'))
        else:
            last_said = 'No, the restaurant servers %s food' %(pref_info.get_value(last_suggested,'food'))
            return(input(last_said + '\n'))
     
    if conf_price != []:
        if conf_price[0] == pref_info.get_value(last_suggested,'pricerange'):
            last_said = 'Yes, the restaurant is in the %s price range' %(pref_info.get_value(last_suggested,'pricerange'))
            return(input(last_said + '\n'))
        else:
            last_said = 'No, the restaurant is in the %s price range' %(pref_info.get_value(last_suggested,'pricerange'))
            return(input(last_said + '\n'))
            
    if conf_location != []:
        if conf_location[0] == pref_info.get_value(last_suggested,'area'):
            last_said = 'Yes, the restaurant is in the %s of town' %(pref_info.get_value(last_suggested,'area'))
            return(input(last_said + '\n'))
        else:
            last_said = 'No, the restaurant is in the %s of town' %(pref_info.get_value(last_suggested,'area'))
            return(input(last_said + '\n'))
               
    last_said = "Sorry, I do not understand what you said"
    return(input(last_said + '\n'))
    
def deny():
    last_said = "What are you exactly looking for? "
    return(input(last_said + '\n'))
    
def give_suggestion(last_suggested):
    # TODO: suggesties moeten alle gespecificeerde voorkeuren (prijs, food en area) in volgorde van opgeven geven.
    last_said = (pref_info.get_value(last_suggested,'restaurantname') + ' is an ' + pref_info.get_value(last_suggested,'pricerance') + ' restaurant serving ' + pref_info.get_value(last_suggested,'food') + ' food in the ' + pref_info.get_value(last_suggested,'area') + ' part of town')
    return(input(last_said + '\n'))
    
      
conversation(speech_act)
