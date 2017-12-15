class Validator:
    def __init__(self, nursery, solution):
        self.nursery = nursery
        self.solution = solution

    def _simulated_annealing_attack_modification(self, is_original=True, old_queen=(), new_queen=()):
        if is_original:
            return self.solution, [1 for x in self.solution]
        else:
            return [old_queen, new_queen], [-2, 2]

    def is_solution_valid(self, calculate_attacks=False, is_original=True, old=(), new=()):

        attacks = 0
        result = True
        iteration_queens, values = self._simulated_annealing_attack_modification(is_original, old, new)
        solution = self.solution
        nursery = self.nursery
        N = len(nursery)

        for i, val in zip(iteration_queens, values):
            row = i[0]
            column = i[1]

            # ROW Check
            for j in xrange(column - 1, -1, -1):
                if (row, j) in solution:
                    attacks += val
                    result &= False
                elif nursery[row][j] == 2:
                    break

            for j in xrange(column + 1, N):
                if (row, j) in solution:
                    attacks += val
                    result &= False
                elif nursery[row][j] == 2:
                    break

            # COLUMN check
            for j in xrange(row - 1, -1, -1):
                if (j, column) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][column] == 2:
                    break

            for j in xrange(row + 1, N):
                if (j, column) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][column] == 2:
                    break

            # ANTIDIAGONAL CHECK
            j = row - 1
            k = column + 1
            while j >= 0 and k < N:
                if (j, k) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][k] == 2:
                    break
                j -= 1
                k += 1

            j = row + 1
            k = column - 1
            while j < N and k >= 0:
                if (j, k) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][k] == 2:
                    break
                j += 1
                k -= 1

            # DIAGONAL CHECK
            j = row - 1
            k = column - 1
            while j >= 0 and k >= 0:
                if (j, k) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][k] == 2:
                    break
                j -= 1
                k -= 1

            j = row + 1
            k = column + 1
            while j < N and k < N:
                if (j, k) in solution:
                    attacks += val
                    result &= False
                elif nursery[j][k] == 2:
                    break
                j += 1
                k += 1

        if calculate_attacks:
            return attacks
        else:
            return result
