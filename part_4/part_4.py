import csv
import random


"""
Print welcome
Recognize speech act of user input
respond according to speech act
"""

# read in restaurant data
restaurant_info = csv.reader(open("restaurantinfo.csv"), delimiter=",")

# train speech act categorization network
# TODO: train network 
# Dit is het eerste deel van part_2

# initialize variables
speech_act = ""
last_said = ""

# start conversation
print('Welcome to the restaurant system. You can ask for restaurants by price, area and the type of food. What would you like?')

while speech_act != "bye" and speech_act != "thankyou":
    # identify speech act
    #TODO: Find speechact: gebruik user input en getrainde netwerk van part 2
    #speech_act = 
    
    # respond to speech act
    respond_to_user(speechact)
    
    
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
    
def request():
    
def thankyou():
    print("You're welcome. Goodbye!")
    exit()
    
def null():
    last_said = "Sorry, I don't understand what you mean"
    print(last_said)
    
def reqalts():

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
    
def restart():
    
def confirm():
    
def deny():
    


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

if speech_act == 'bye':
    exit()




