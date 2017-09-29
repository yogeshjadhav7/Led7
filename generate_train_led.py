import numpy as np
from copy import deepcopy

class Generate(object):
    def __init__(self):
        self.content = "A,B,C,D,E,F,G,AB,AC,AD,AE,AF,AG,BC,BD,BE,BF,BG,CD,CE,CF,CG,DE,DF,DG,EF,EG,FG,0,1,2,3,4,5,6,7,8,9\n";
        self.led = [[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1],
             [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1],
             [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1]]

        self.generate()

    def generate(self):
        for i in range(np.alen(self.led)):
            length = len(self.led[i])
            state = ""
            for j in range(length):
                if state == "":
                    state = str(self.led[i][j])
                else:
                    state = state + "," + str(self.led[i][j])

            state = self.create_extra_features(state)
            state_resp = np.zeros(10, np.int32)
            state_resp[i] = 1
            for j in range(10):
                state = state + "," + str(state_resp[j])

            state = state + "\n"
            self.content = self.content + state



    def create_extra_features(self, state):
        length = len(state)
        for i in range(length):
            if state[i] == ',':
                continue

            j = i + 1
            while j < length:
                if state[j] == ',':
                    j = j + 1
                    continue

                if state[i] == '0' or state[j] == '0':
                    state = state + ",0"
                else:
                    state = state + ",1"

                j = j + 1

        return state


output_file_name = "Train-LED.csv"
gen = Generate()
content = gen.content
f = open(output_file_name, 'w')
f.write(content)
f.close()