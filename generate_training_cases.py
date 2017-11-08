import numpy as np

supervised_data = []
supervised_data.append(("1111110", 0))
supervised_data.append(("0110000", 1))
supervised_data.append(("1101101", 2))
supervised_data.append(("1111001", 3))
supervised_data.append(("0110011", 4))
supervised_data.append(("1011011", 5))
supervised_data.append(("1011111", 6))
supervised_data.append(("1110000", 7))
supervised_data.append(("1111111", 8))
supervised_data.append(("1111011", 9))
supervised_data.append(("1110001", 3))
supervised_data.append(("1111000", 3))
supervised_data.append(("0111111", 8))
supervised_data.append(("1001111", 6))
supervised_data.append(("0110110", 0))
supervised_data.append(("0010000", 1))
supervised_data.append(("0010000", 1))
supervised_data.append(("0100011", 4))
supervised_data.append(("0010011", 4))
supervised_data.append(("0000110", 6))
supervised_data.append(("0100101", 4))
supervised_data.append(("0100101", 2))
supervised_data.append(("1100100", 2))
supervised_data.append(("1110100", 0))
supervised_data.append(("0111100", 0))
supervised_data.append(("0110100", 0))
supervised_data.append(("1010011", 5))
supervised_data.append(("1111101", 8))
supervised_data.append(("0110101", 8))
supervised_data.append(("0110101", 8))
supervised_data.append(("1110011", 9))


class Generate(object):
    def __init__(self):
        self.content = "A,B,C,D,E,F,G,AB,AC,AD,AE,AF,AG,BC,BD,BE,BF,BG,CD,CE,CF,CG,DE,DF,DG,EF,EG,FG\n"
        self.generate()

    def generate(self):
        length = len(supervised_data)
        for i in range(length):
            local_content = ""
            state_unprep = supervised_data[i][0]
            state = ""

            for j in range(7):
                if state == "":
                    state = state_unprep[j]
                else:
                    state = state + "," + state_unprep[j]

            label = supervised_data[i][1]
            local_content = local_content + self.create_extra_features(state)
            labels = np.zeros(10, np.int32)
            labels[label] = 1
            for j in range(10):
                local_content = local_content + "," + str(labels[j])

            self.content = self.content + local_content + "\n"

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