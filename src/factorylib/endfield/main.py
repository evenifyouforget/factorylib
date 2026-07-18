import numpy as np
from numpy import inf
from scipy.optimize import Bounds, LinearConstraint, milp

_unique_counter = 0

SOLID = 'S'
LIQUID = 'L'
GAS = 'G'
VIRTUAL = 'V'

class Material(object):
    def __init__(self, _id=None, name=None, unit='', tags=None):
        global _unique_counter
        if _id is None:
            _id = _unique_counter
            _unique_counter += 1
        if name is None:
            name = f"Anonymous Material #{_unique_counter}"
        self._id = _id
        self.name = name
        self.unit = unit
        self.tags = tags
    def __eq__(self, other):
        return isinstance(other, Material) and self._id == other._id
    def __hash__(self):
        return hash((self._id, 12345)) # magic number
    def substitute(self, subs_dict):
        return subs_dict[self]
    def __add__(self, other):
        return AddMaterial(self, other)
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + other * -1
    def __rsub__(self, other):
        return other + self * -1
    def __mul__(self, other):
        return MulMaterial(self, other)
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * -1

class AddMaterial(Material):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def substitute(self, subs_dict):
        return substitute(self.lhs, subs_dict) + substitute(self.rhs, subs_dict)

class MulMaterial(Material):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def substitute(self, subs_dict):
        return substitute(self.lhs, subs_dict) * substitute(self.rhs, subs_dict)

def substitute(expr, subs_dict):
    if isinstance(expr, (int, float)):
        return expr
    return expr.substitute(subs_dict)

class Recipe(object):
    def __init__(self, expression, name, max_multiples=inf, integer_only=False):
        self.expression = expression
        self.name = name
        self.max_multiples = max_multiples
        self.integer_only = integer_only

def optimize(all_materials, all_recipes, material_to_maximize):
    # Get all recipes
    num_recipes = len(all_recipes)
    # Get the complete list of all materials, including recipe counters
    all_materials = list(set(all_materials))
    num_materials = len(all_materials)
    # Assign each material a unit basis vector
    subs_dict = {}
    max_objective_index = "err"
    for i, material in enumerate(all_materials):
        a = np.zeros(num_materials, dtype=float)
        a[i] = 1
        subs_dict[material] = a
        if material == material_to_maximize:
            max_objective_index = i
    # Construct the recipe matrix
    recipe_matrix = np.zeros((num_recipes, num_materials), dtype=float)
    for i, recipe in enumerate(all_recipes):
        recipe_matrix[i,:] += substitute(recipe.expression, subs_dict)
    # Construct the bounds on the decision variables (recipe multiples)
    lb = np.zeros(num_recipes, dtype=float)
    ub = np.full(num_recipes, inf, dtype=float)
    for i, recipe in enumerate(all_recipes):
        ub[i] = recipe.max_multiples
    bounds = Bounds(lb=lb, ub=ub)
    # Construct the integrality flags
    integrality = np.zeros(num_recipes, dtype=int)
    for i, recipe in enumerate(all_recipes):
        if recipe.integer_only:
            integrality[i] = 1
    # Construct the constraints (all net supplies must be non-negative)
    constraints = LinearConstraint(recipe_matrix.T, lb=0)
    # Minimization objective
    c = -recipe_matrix[:,max_objective_index]
    # Query MILP solver
    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    # Print result
    print(f'# Result {res.status}: ' + res.message)
    print(f'- Maximized score: {-res.fun}')
    print('## Recipes Used')
    for i, multiples in enumerate(res.x):
        if multiples == 0:
            continue
        print(f'- {multiples} multiples of {all_recipes[i].name}')
    print('## Balance Sheet Per Material')
    net_amount = recipe_matrix.T @ res.x
    bits = [[f'### {material.name} (net {net}{material.unit})'] for material, net in zip(all_materials, net_amount)]
    for i, multiples in enumerate(res.x):
        if multiples == 0:
            continue
        recipe = all_recipes[i]
        for j, material in enumerate(all_materials):
            per_multiple = recipe_matrix[i,j]
            if per_multiple == 0:
                continue
            contribution = multiples * per_multiple
            bits[j].append(f'- {contribution}{material.unit} from {multiples} multiples of {recipe.name}')
    print('\n'.join(map('\n'.join, bits)))

def main():
    FOOIUM = Material(name='Fooium', unit='/min', tags=SOLID)
    BARIUM = Material(name='Barium', unit='/min', tags=SOLID)
    FOOBARIUM = Material(name='Foobarium', unit='/min', tags=SOLID)
    all_materials = [FOOIUM, BARIUM, FOOBARIUM]
    FREE_MATERIALS = Recipe(FOOIUM * 3 + BARIUM * 7, name="Starting Materials", max_multiples=1)
    FOO_PLUS_BAR = Recipe(-FOOIUM - BARIUM + FOOBARIUM, name="Add Foo And Bar")
    SPECIAL_OFFER = Recipe(-2 * FOOIUM - BARIUM + 4 * FOOBARIUM, name="Special Integer Reaction", max_multiples=2, integer_only=1)
    PURE_BARIUM = Recipe(-BARIUM + 0.1 * FOOBARIUM, name="Inefficient Barium Conversion")
    all_recipes = [FREE_MATERIALS, FOO_PLUS_BAR, SPECIAL_OFFER, PURE_BARIUM]
    optimize(all_materials, all_recipes, FOOBARIUM)