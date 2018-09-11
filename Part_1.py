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

train_test = ["dstc2_train", "dstc2_test"]

with open("dialogues.txt", "w") as train:
    for f_name in train_test:
        train_path = f_name + "/data"
        train_dirs = os.listdir(train_path)
        for train_dir in train_dirs:
            data_path = train_path + "/" + train_dir
            data_dirs = os.listdir(data_path)
            for data_dir in data_dirs:
                s_file = open(data_path + "/" + data_dir + "/log.json").read()
                s_dict = json.loads(s_file)
                s_turns = s_dict['turns']

                u_file = open(data_path + "/" + data_dir + "/label.json").read()
                u_dict = json.loads(u_file)
                u_turns = u_dict['turns']
                u_taskinfo = u_dict['task-information']
                u_goal = u_taskinfo['goal']
                u_task = u_goal['text']
                print(f"session id: {data_dir}", file=train)
                print(f"{u_task}", file=train)


                for i in range(0,max(len(s_turns),len(u_turns))):
                    s_turn = s_dict['turns'][i]
                    s_index = s_turn['turn-index']
                    s_output = s_turn['output']
                    s_utt = s_output['transcript']
                    print(f"system: {s_utt}", file=train)

                    u_turn = u_dict['turns'][i]
                    u_index = u_turn['turn-index']
                    u_utt = u_turn['transcription']
                    print(f"user: {u_utt}", file=train)
