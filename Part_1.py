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

path_to_project = "C:/Users/tycho/OneDrive/Documenten/GitHub/MAIR_Team_Project/" # please fill in path to project folder
train_test = ["dstc2_train", "dstc2_test"] # names of the folders with the train
# and test data

with open("dialogues.txt", "w") as dial:
    for f_name in train_test: # For the folder with train and the folder with
    # test-data:
        f_path = path_to_project + f_name + "/data" # Save the path to the data folder
        f_dirs = os.listdir(f_path) # Save the names of all folders in the
        # training and test folders (the "Mar13...") folders.
        for f_dir in f_dirs:
            data_path = f_path + "/" + f_dir # Save the path to each of the
            # "Mar13..." folders.
            data_dirs = os.listdir(data_path) # Save the names of each of the dialogue
            # folders in the "Mar13..." folders.
            for data_dir in data_dirs:
                s_file = open(data_path + "/" + data_dir + "/log.json").read()
                # path to system dialogue info
                s_dict = json.loads(s_file) # save the jason file in a Python
                # dictionary
                s_turns = s_dict['turns']

                # Do the same for the user, additionally, save the task and
                # the session id (folder name)
                u_file = open(data_path + "/" + data_dir + "/label.json").read()
                u_dict = json.loads(u_file)
                u_turns = u_dict['turns']
                u_taskinfo = u_dict['task-information']
                u_goal = u_taskinfo['goal']
                u_task = u_goal['text']
                print(f"session id: {data_dir}", file=dial)
                print(f"{u_task}", file=dial)
                print()
                print("session id: " + data_dir)
                print(u_task )

                for i in range(0,len(s_turns)):
# The above range only makes sense if the number of system turns always equals
# the number of user turns. I think it is the case, but I did not check.
                    # Print user and system utterances to dialogues file.
                    s_turn = s_dict['turns'][i]
                    s_index = s_turn['turn-index']
                    s_output = s_turn['output']
                    s_utt = s_output['transcript']
                    print(f"system: {s_utt}", file=dial)
                    print("system:" + s_utt)

                    u_turn = u_dict['turns'][i]
                    u_index = u_turn['turn-index']
                    u_utt = u_turn['transcription']
                    print(f"user: {u_utt}", file=dial)
                    print("user: " + u_utt)

                input("Please press enter for the next dialogue...")
