import numpy as np
from copy import deepcopy

class Generate(object):
    def __init__(self):
        self.content = "A,B,C,D,E,F,G\n";
        self.generate("", 1)

    def generate(self, s, n):
        if n == 7:
            self.content = self.content + s + "0\n"
            self.content = self.content + s + "1\n"
            return

        self.generate(s + "0,", n + 1)
        self.generate(s + "1,", n + 1)

output_file_name = "all_cases.csv"
gen = Generate()
content = gen.content
f = open(output_file_name, 'w')
f.write(content)
f.close()