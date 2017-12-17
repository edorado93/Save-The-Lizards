# The Lizard Runner Algorithmic implementation with trees and test cases included.

from functools import wraps
import time
from Validator import Validator
import optparse


def get_options():
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="input file containing initial config.")
    parser.add_option("-s", "--solution", action="store_true", dest="should_display_sol", help="Display entire solution instead of OK/FAIL")
    return parser.parse_args()[0]


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time() * 1000
        result = func(*args, **kwargs)
        end = time.time() * 1000
        print "Time taken for function = {} is {} ms".format(func.__name__, end - start)
        return result

    return wrapper


class Zoo:
    def __init__(self, nursery):
        self.nursery = nursery
        self.tree_locations = {}
        self.rows = {}
        self.columns = {}
        self.diagonals = {}
        self.antidiagonals = {}
        self.did_tree_have_masking_effect = {}
        self.is_there_queen_in_this_column = {}
        # Pre-processing logic for populating the trees
        self.next_position_same_column = []
        self.is_there_a_tree_ahead = {}
        self.preprocess()

    @staticmethod
    def hash_util(dictionary, key, value_to_add, is_marking, is_tree):
        result = 1
        if is_marking:
            if is_tree:
                if key not in dictionary or dictionary[key] < 0:
                    result = -1  # Do nothing
                else:
                    dictionary[key] = value_to_add  # There was a queen here
            else:
                dictionary[key] = value_to_add
                result = -1
        else:
            dictionary[key] = value_to_add

        return result

    def get_nursery(self):
        return self.nursery

    def set_nursery(self, nursery):
        self.nursery = nursery

    def mark_visited(self, row, col, is_tree=False):
        val = -1 if is_tree else 1
        a = Zoo.hash_util(self.rows, row, val, True, is_tree)
        b = Zoo.hash_util(self.columns, col, val, True, is_tree)
        c = Zoo.hash_util(self.diagonals, row - col, val, True, is_tree)
        d = Zoo.hash_util(self.antidiagonals, row + col, val, True, is_tree)
        if is_tree:
            self.did_tree_have_masking_effect[(row, col)] = (a, b, c, d)

    def unmark_visited(self, row, col, is_tree=False):
        if is_tree:
            # Tree
            val = self.did_tree_have_masking_effect[(row, col)]
            del self.did_tree_have_masking_effect[(row, col)]
        else:
            # Queen
            val = (-1, -1, -1, -1)
        Zoo.hash_util(self.rows, row, val[0], False, is_tree)
        Zoo.hash_util(self.columns, col, val[1], False, is_tree)
        Zoo.hash_util(self.diagonals, row - col, val[2], False, is_tree)
        Zoo.hash_util(self.antidiagonals, row + col, val[3], False, is_tree)

    def trees_populator(self):
        nursery = self.nursery
        row = len(nursery)
        col = len(nursery[0])
        for i in xrange(0, row):
            for j in xrange(0, col):
                # We found a tree
                if nursery[i][j] == 2:
                    self.tree_locations[(i, j)] = 1
                    for k in xrange(0, j+1):
                        self.is_there_a_tree_ahead[k] = 1


    def find_next_largest(self, column_number):
        length = len(self.nursery) - 1
        stack = [length]
        next_lar = [-1]
        top = 0
        length -= 1
        while length >= 0:
            while stack and self.nursery[length][column_number] >= self.nursery[stack[top]][column_number]:
                stack.pop()
                top -= 1
            if top == -1:
                next_lar.append(-1)
            else:
                next_lar.append(stack[top])
            stack.append(length)
            top += 1

            length -= 1
        return next_lar

    def preprocess(self):
        self.trees_populator()
        nursery = self.nursery
        length = len(nursery)
        next_largest = []

        for i in xrange(0, length):
            ans = self.find_next_largest(i)
            next_largest.append(ans[::-1])
        self.next_position_same_column = zip(*next_largest)

    """
        Returns True if we can successfully place a queen at the given
        row and column. False othrwise. O(1) complexity 
    """

    def is_cell_safe(self, row, column):
        if row in self.rows and self.rows[row] > 0:
            return False
        if column in self.columns and self.columns[column] > 0:
            return False
        if row - column in self.diagonals and self.diagonals[row - column] > 0:
            return False
        if row + column in self.antidiagonals and self.antidiagonals[row + column] > 0:
            return False
        return True

    def print_data_structures(self):
        return "rows = {}, columns = {}, diag = {}, anti = {}".format(self.rows, self.columns, self.diagonals,
                                                                      self.antidiagonals)


class DFS:
    def __init__(self, zoo, number_of_lizards_to_place, args):
        self.zoo = zoo
        self.lizards_to_place = number_of_lizards_to_place
        self.solution = set()
        self.N = len(zoo.get_nursery())
        self.should_display_sol = args.should_display_sol

    def is_cell_invalid(self, row, col):
        N = self.N
        return row >= N or col >= N

    @timer
    def run(self, row, column, lizards_successfully_placed):
        return self.dfs(row, column, lizards_successfully_placed)

    def dfs(self, row, column, lizards_successfully_placed):

        zoo = self.zoo
        n = self.N

        if lizards_successfully_placed == self.lizards_to_place:
            if self.should_display_sol:
                print "Solution found {} and is_valid = {}".format(self.solution, Validator(self.zoo.get_nursery(),
                                                                                              self.solution).is_solution_valid())
            return True

        # We found a dead end with no solution
        if self.is_cell_invalid(row, column):
            return False

        is_tree = (row, column) in zoo.tree_locations
        if is_tree:
            # Will mark a tree and make it anti queen
            zoo.mark_visited(row, column, True)

        dfs_result = False
        if not is_tree and zoo.is_cell_safe(row, column):
            # Mark the current cell as visited and add it to the solution
            zoo.mark_visited(row, column, False)
            self.solution.add((row, column))
            if column in zoo.is_there_queen_in_this_column:
                zoo.is_there_queen_in_this_column[column] += 1
            else:
                zoo.is_there_queen_in_this_column[column] = 1

            next_row_number_in_same_column = self.zoo.next_position_same_column[row][column]
            if next_row_number_in_same_column != -1 and next_row_number_in_same_column < n:
                dfs_result = self.dfs(next_row_number_in_same_column, column, lizards_successfully_placed + 1)
            else:
                dfs_result = self.dfs(0, column + 1, lizards_successfully_placed + 1)

            if dfs_result:
                return True

            # Unmark the current cell and remove it from the solution as well
            zoo.unmark_visited(row, column, False)
            self.solution.remove((row, column))
            if column in zoo.is_there_queen_in_this_column:
                zoo.is_there_queen_in_this_column[column] -= 1

        # Only recurse further if the solution wasn't found in the previous recursions
        if not dfs_result:
            if row + 1 < n:
                dfs_result = self.dfs(row + 1, column, lizards_successfully_placed)
            else:
                if column in zoo.is_there_queen_in_this_column and zoo.is_there_queen_in_this_column[column] == 0 and column not in zoo.is_there_a_tree_ahead and (self.lizards_to_place - lizards_successfully_placed) >= (n - column + 1):
                    pass
                else:
                    dfs_result = self.dfs(0, column + 1, lizards_successfully_placed)

        if is_tree:
            # Will un-mark a tree and remove its effect
            zoo.unmark_visited(row, column, True)

        return dfs_result


if __name__ == "__main__":
    args = get_options()
    filename = args.filename
    line_number = 1
    zoo = []
    with open(filename) as input_file:
        for line in input_file:
            line = line.strip()
            if line_number == 1:
                n = int(line)
            elif line_number == 2:
                number_of_lizards = int(line)
            elif line_number < (n + 2):
                zoo.append(map(int, list(line)))
            else:
                zoo.append(map(int, list(line)))
                line_number = 0  # resetting for the next input
                result = DFS(Zoo(zoo), number_of_lizards, args).run(0, 0, 0)
                if not result:
                    print("FAIL")
                else:
                    print('OK')
                zoo = []
            line_number += 1
