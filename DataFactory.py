import numpy as np
from math import sqrt,log
import random


class DataFactory():

    def __init__(self, file_path, batch_size, n_classes):
        self.filePath = file_path
        self.game_state_move_dict = eval(open("state_move_dict.txt", 'r').read())
        self.game_states = self.game_state_move_dict.keys()
        self.batch_counter = 0
        self.n_classes = n_classes
        self.batch_size = batch_size

    def next_batch(self):
        start_index = self.batch_counter * self.batch_size
        end_index = start_index + self.batch_size - 1
        states = []
        moves = []

        for index, state in enumerate(self.game_states):
            if index < start_index:
                continue

            if index > end_index:
                break

            each_state = self._get_normalized_state(state)

            move = self.game_state_move_dict.get(state)
            each_move = np.zeros(4)
            each_move[move] = 1

            if len(states) == 0:
                states = each_state
                moves = each_move
            else:
                if random.randint(1, 2) % 2 != 0:
                    states = np.vstack((states, each_state))
                    moves = np.vstack((moves, each_move))
                else:
                    states = np.vstack((each_state, states))
                    moves = np.vstack((each_move, moves))

        self.batch_counter = self.batch_counter + 1

        #print (states)
        #print (moves)
        return states, moves

    def is_empty(self):
        start_index = self.batch_counter * self.batch_size
        if start_index > len(self.game_states):
            return True

        return False

    def get_all_game_states_moves(self):
        states = []
        moves = []
        for index, state in enumerate(self.game_states):
            #if random.randint(1, 3) % 3 != 0:
             #   continue

            each_state = self._get_normalized_state(np.matrix(str(state)).flatten())
            move = self.game_state_move_dict.get(state)
            each_move = np.zeros(4)
            each_move[move] = 1

            if len(states) == 0:
                states = each_state
                moves = each_move
            else:
                if random.randint(1, 2) % 2 != 0:
                    states = np.vstack((states, each_state))
                    moves = np.vstack((moves, each_move))
                else:
                    states = np.vstack((each_state, states))
                    moves = np.vstack((each_move, moves))

        return states, moves

    def _get_normalized_state(self, state):
        processed_state = self._process_state(state)
        n_state = np.zeros(len(processed_state), np.float64)

        low_offset = 12
        for index in range(len(processed_state)):
            if processed_state[index] != 0:
                n_state[index] = np.log2(float(processed_state[index]))
                if n_state[index] < low_offset:
                    low_offset = n_state[index]

        #for index in range(len(n_state)):
         #   if n_state[index] != 0:
          #      n_state[index] = n_state[index] - low_offset + 1

        highest_cell_value = 10
        n_state = np.divide(n_state, highest_cell_value)
        n_state = np.square(n_state)
        #print (state)
        #print (n_state)
        #print ("\n\n")
        return n_state

    def _process_state(self, state):

        state_matrix = str(np.matrix(str(state)))
        state_matrix = state_matrix.replace('[', '').replace(']', '').replace('\n', '').split(' ')
        state_matrix_len = len(state_matrix)
        processed_state = []

        for index in range(state_matrix_len):
            if state_matrix[index].isdigit():
                processed_state.append(int(state_matrix[index]))

        return processed_state
