# The Lizard Runner Algorithmic implementation with trees and test cases included.

from functools import wraps
import time
import copy
from Validator import Validator

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time() * 1000
        result = func(*args, **kwargs)
        end = time.time() * 1000
        print "Time taken for function = {} is {} ms".format(func.__name__, end - start)
        return result

    return wrapper

class BasicNQueens:

    def __init__(self, matrix, lizards, size, ans):
        self.zoo = matrix
        self.lizards = lizards
        self.size = size
        self.all_ones = (1 << self.size) - 1
        self.number_of_solutions = 0
        self.solution_board = []
        self.ans = set()

    def run(self):
        self.dfs_runner(0, 0, 0, 0)
        return self.number_of_solutions

    def dfs_runner(self, column, left_diagonal, right_diagonal, queens_placed):
        if self.number_of_solutions == 1:
            return True
        # Need to make this more efficient. This slows down by a factor of 2
        if queens_placed == self.lizards:
            self.number_of_solutions += 1
            self.ans = copy.deepcopy(self.solution_board)

        valid_spots = self.all_ones & ~(column | left_diagonal | right_diagonal)
        while valid_spots != 0:
            current_spot = -valid_spots & valid_spots
            self.solution_board.append((current_spot & -current_spot).bit_length() - 1)
            valid_spots ^= current_spot
            result = self.dfs_runner((column | current_spot), (left_diagonal | current_spot) >> 1,
                            (right_diagonal | current_spot) << 1, queens_placed + 1)
            if result:
                return True

            self.solution_board.pop()

        return False


class Zoo:
    def __init__(self, nursery, should_preprocess=True):
        self.nursery = nursery
        self.tree_locations = {}
        self.rows = {}
        self.columns = {}
        self.diagonals = {}
        self.antidiagonals = {}
        self.did_tree_affect = {}
        self.next_largest = []
        self.is_there_anything_in_this_column = {}
        self.is_there_a_tree = {}
        if should_preprocess:
            self.preprocess()

    @staticmethod
    def hash_util(hashwa, key, value_to_add, is_marking, is_tree):
        result = 1
        if is_marking:
            if is_tree:
                if key not in hashwa or hashwa[key] < 0:
                    result = -1  # Do nothing
                else:
                    hashwa[key] = value_to_add  # There was a queen here
            else:
                hashwa[key] = value_to_add
                result = -1
        else:
            hashwa[key] = value_to_add

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
            self.did_tree_affect[(row, col)] = (a, b, c, d)

    def unmark_visited(self, row, col, is_tree=False):

        if is_tree:
            # Tree
            val = self.did_tree_affect[(row, col)]
            del self.did_tree_affect[(row, col)]
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
                        self.is_there_a_tree[k] = 1

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
        self.next_largest = zip(*next_largest)

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
    def __init__(self, zoo, number_of_lizards_to_place):
        self.zoo = zoo
        self.lizards_to_place = number_of_lizards_to_place
        self.N = len(zoo.get_nursery())
        self.stack = []
        self.solution = set()

    def is_cell_invalid(self, row, col):
        N = self.N
        return row >= N or col >= N

    @timer
    def run(self):
        if len(self.zoo.tree_locations) == 0:
            solution = self.solution
            nqueens = BasicNQueens(self.zoo.get_nursery(), self.lizards_to_place, self.N, self.solution)
            result = nqueens.run() == 1
            for i, j in enumerate(nqueens.ans):
                solution.add((i, j))
            return result
        else:
            return self.dfs()

    def dfs(self):

        PARENT = 1
        CHILD = 0
        n = self.N
        stack = self.stack
        solution = self.solution
        zoo = self.zoo
        stack.append((0, 0, 0, CHILD))
        while stack:
            row, column, lizards_successfully_placed, type_of_node = stack.pop()

            is_tree = (row, column) in zoo.tree_locations
            if type_of_node == CHILD:
                if lizards_successfully_placed == self.lizards_to_place:
                    return True

                # We found a dead end with no solution
                if self.is_cell_invalid(row, column):
                    continue

                if is_tree:
                    # Will mark a tree and make it anti queen
                    stack.append((row, column, lizards_successfully_placed, PARENT))
                    zoo.mark_visited(row, column, True)

                if row + 1 < n:
                    stack.append((row + 1, column, lizards_successfully_placed, CHILD))
                else:
                    if column in zoo.is_there_anything_in_this_column and zoo.is_there_anything_in_this_column[column] == 0 and column not in zoo.is_there_a_tree and (self.lizards_to_place - lizards_successfully_placed) >= (n - column + 1):
                        pass
                    else:
                        stack.append((0, column + 1, lizards_successfully_placed, CHILD))

                if not is_tree and zoo.is_cell_safe(row, column):

                    stack.append((row, column, lizards_successfully_placed, PARENT))
                    # Mark the current cell as visited and add it to the solution
                    zoo.mark_visited(row, column, False)
                    solution.add((row, column))
                    if column in zoo.is_there_anything_in_this_column:
                        zoo.is_there_anything_in_this_column[column] += 1
                    else:
                        zoo.is_there_anything_in_this_column[column] = 1

                    next_row_number_in_same_column = self.zoo.next_largest[row][column]
                    if next_row_number_in_same_column != -1 and next_row_number_in_same_column < n:
                        stack.append((next_row_number_in_same_column, column, lizards_successfully_placed + 1, CHILD))
                    else:
                        stack.append((0, column + 1, lizards_successfully_placed + 1, CHILD))

            elif type_of_node == PARENT:
                # Unmark the current cell and remove it from the solution as well
                if is_tree:
                    zoo.unmark_visited(row, column, True)
                if (row, column) in solution:
                    # print "Popping {},{}".format(row, column)
                    solution.remove((row, column))
                    zoo.unmark_visited(row, column, False)
                    if column in zoo.is_there_anything_in_this_column:
                        zoo.is_there_anything_in_this_column[column] -= 1

        return False

def print_solution(solution, trees, n, output):

    for i in xrange(0, n):
        lst = []
        for j in xrange(0, n):
            if (i, j) in solution:
               lst.append('1')
            elif (i, j) in trees:
                lst.append('2')
            else:
                lst.append('0')

        output.write(''.join(lst))
        output.write("\n")

if __name__ == "__main__":
    # filename = "/Users/sachinmalhotra/Documents/USC-Studies/AI-CSCI-560/Solutions/big_input.txt"
    filename = "input.txt"
    line_number = 1
    zoo = []
    output = open("output.txt", "w")
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
                dfs = DFS(Zoo(zoo), number_of_lizards)
                result = dfs.run()

                if not result:
                    output.write("FAIL\n")
                else:
                    output.write("OK\n")
                    #print_solution(dfs.solution, dfs.zoo.tree_locations, dfs.N, output)
                    # print "Solution found {} and is_valid = {}".format(dfs.solution,
                    #                                                     Validator(dfs.zoo.get_nursery(),
                    #                                                               dfs.solution).is_solution_valid())
                zoo = []
            line_number += 1
    output.close()
