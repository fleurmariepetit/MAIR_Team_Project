# General information assignment:
    # 4 main parts:
        # 1. Data preprocessing and modeling
            # Data in form of collection of json files.
            # Write small programm in Python.
                # Read data files
                # Produce dialogue that is defined by them.
            # Build a domain model
                # Make a diagram of the dialogue flow
                # Later part: implement this model.
                # Model successful dialogues.
        # 2. Machine learning
            # Label turns with class from dialogue acts classes.
        # 3. Dialogue
            # Implement categorial grammar.
        # 4. Management
            # Implement dialogue model. Put everything together.

import json
import os
import re

train_test = ["dstc2_train", "dstc2_test"] # Folders with test and train data.

with open("acts_and_utts.txt", "w") as acts_utts: # Open the text file for all dialogues
    for f_name in train_test: # For the folders with test and train data:
        f_path = f_name + "/data" # The path to the data folder in the test and
        # the train folders.
        f_dirs = os.listdir(f_path) # The names of the different test and train
        # sets. These are the folders starting with "Mar13".
        for f_dir in f_dirs: # For each of the "Mar13" folders:
            data_path = f_path + "/" + f_dir # Define the path to the folder.
            data_dirs = os.listdir(data_path) # The names of all single dialogue files
            # in a "Mar13" folder.
            for data_dir in data_dirs: # For each single dialogue:
                u_file = open(data_path + "/" + data_dir + "/label.json").read()
                u_dict = json.loads(u_file)
                u_turns = u_dict['turns']
                for i in range(0,len(u_turns)): # for each turn do:
# The above range only makes sense if the number of system turns always equals
# the number of user turns. I think it is the case, but I did not check.
                    u_turn = u_dict['turns'][i]
                    u_index = u_turn['turn-index']
                    u_cam = u_turn['semantics']['cam'] # the speech act + values
                    u_act = re.match('^[a-z]+', u_cam).group(0) # match speech act type only
                    u_utt = u_turn['transcription'] # save user utterance
                    print(f"{u_act} {u_utt}", file=acts_utts) # print user utterance to
                    # text file with all dialogues.
                    print(f"{u_act} {u_utt}")
