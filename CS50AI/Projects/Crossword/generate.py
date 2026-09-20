import sys
from crossword import Crossword, Variable


class CrosswordCreator:

    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, then run backtracking search.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update self.domains such that each variable is node-consistent.
        Remove words that do not match the variable's length.
        """
        for var in self.domains:
            invalid_words = {word for word in self.domains[var] if len(word) != var.length}
            self.domains[var] -= invalid_words

    def revise(self, x, y):
        """
        Make variable x arc consistent with variable y.
        Remove values from self.domains[x] for which there is no possible
        corresponding value for y in self.domains[y].
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        i, j = overlap
        stale_words = set()

        for word_x in self.domains[x]:
            # Check if any word in y's domain matches character at overlap
            if not any(word_x[i] == word_y[j] for word_y in self.domains[y]):
                stale_words.add(word_x)
                revised = True

        self.domains[x] -= stale_words
        return revised

    def ac3(self, arcs=None):
        """
        Update self.domains such that each variable is arc consistent.
        """
        if arcs is None:
            # Build initial queue of all arc pairs between overlapping neighbors
            queue = [
                (x, y) for x in self.domains
                for y in self.crossword.neighbors(x)
            ]
        else:
            queue = list(arcs)

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if assignment is complete (i.e., assigns a value to each variable).
        """
        return len(assignment) == len(self.crossword.variables)

    def consistent(self, assignment):
        """
        Return True if assignment is consistent (values fit lengths,
        are unique, and satisfy all overlapping character constraints).
        """
        # Ensure all assigned values are unique
        assigned_words = list(assignment.values())
        if len(assigned_words) != len(set(assigned_words)):
            return False

        for var, word in assignment.items():
            # Ensure correct length
            if len(word) != var.length:
                return False

            # Check character overlaps with assigned neighbors
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap:
                        i, j = overlap
                        if word[i] != assignment[neighbor][j]:
                            return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of var, ordered by
        the least-constraining values heuristic.
        """
        unassigned_neighbors = self.crossword.neighbors(var) - set(assignment.keys())

        def count_eliminations(val):
            count = 0
            for neighbor in unassigned_neighbors:
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap:
                    i, j = overlap
                    for neighbor_word in self.domains[neighbor]:
                        if val[i] != neighbor_word[j]:
                            count += 1
            return count

        return sorted(self.domains[var], key=count_eliminations)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable using the Minimum Remaining Values (MRV)
        heuristic, breaking ties with the Degree Heuristic.
        """
        unassigned = [var for var in self.crossword.variables if var not in assignment]

        # Sort by domain size (MRV), then highest degree (number of neighbors)
        return min(
            unassigned,
            key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var)))
        )

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete satisfying assignment if possible.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            test_assignment = assignment.copy()
            test_assignment[var] = value

            if self.consistent(test_assignment):
                result = self.backtrack(test_assignment)
                if result is not None:
                    return result

        return None


def main():

    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "_main_":
    main()
