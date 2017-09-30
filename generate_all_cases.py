import numpy as np
from copy import deepcopy

class Generate(object):
    def __init__(self):
        self.content = "A,B,C,D,E,F,G,AB,AC,AD,AE,AF,AG,BC,BD,BE,BF,BG,CD,CE,CF,CG,DE,DF,DG,EF,EG,FG\n";
        self.generate("", 1)

    def generate(self, s, n):
        if n == 7:
            self.content = self.content + self.create_extra_features(s + "0") + "\n"
            self.content = self.content + self.create_extra_features(s + "1") + "\n"
            return

        self.generate(s + "0,", n + 1)
        self.generate(s + "1,", n + 1)


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

output_file_name = "all_cases.csv"
gen = Generate()
content = gen.content
f = open(output_file_name, 'w')
f.write(content)
f.close()