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
    def __lt__(self, other):
        return self.name < other.name
    def __le__(self, other):
        return self.name <= other.name
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

def test_main():
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

def main():
    PMIN = '/min'
    WulingStockBill = Material(name='$', unit=PMIN, tags=VIRTUAL)
    Watt = Material(name='W', tags=VIRTUAL)
    def std_solid(name):
        return Material(name=name, unit=PMIN, tags=SOLID)
    def std_liquid(name):
        return Material(name=name, unit=PMIN, tags=LIQUID)
    def std_gas(name):
        return Material(name=name, unit=PMIN, tags=GAS)
    OriginiumOre = std_solid('Originium Ore')
    AmethystOre = std_solid('Amethyst Ore')
    FerriumOre = std_solid('Ferrium Ore')
    CupriumOre = std_solid('Cuprium Ore')
    Buckflower = std_solid('Buckflower')
    Citrome = std_solid('Citrome')
    Aketine = std_solid('Aketine')
    Sandleaf = std_solid('Sandleaf')
    Yazhen = std_solid('Yazhen')
    Jincao = std_solid('Jincao')
    BuckflowerPowder = std_solid('Buckflower Powder')
    CitromePowder = std_solid('Citrome Powder')
    AketinePowder = std_solid('Aketine Powder')
    SandleafPowder = std_solid('Sandleaf Powder')
    YazhenPowder = std_solid('Yazhen Powder')
    JincaoPowder = std_solid('Jincao Powder')
    Cuprium = std_solid('Cuprium')
    CupriumPowder = std_solid('Cuprium Powder')
    Ferrium = std_solid('Ferrium')
    FerriumPowder = std_solid('Ferrium Powder')
    Amethyst = std_solid('Amethyst')
    AmethystPowder = std_solid('Amethyst Powder')
    OriginiumPowder = std_solid('Originium Powder')
    Carbon = std_solid('Carbon')
    CarbonPowder = std_solid('Carbon Powder')
    Origocrust = std_solid('Origocrust')
    OrigocrustPowder = std_solid('Origocrust Powder')
    DenseOrigocrustPowder = std_solid('Dense Origocrust Powder')
    PackedOrigocrust = std_solid('Packed Origocrust')
    DenseOriginiumPowder = std_solid('Dense Originium Powder')
    DenseFerriumPowder = std_solid('Dense Ferrium Powder')
    Steel = std_solid('Steel')
    CrystonPowder = std_solid('Cryston Powder')
    CrystonFiber = std_solid('Cryston Fiber')
    DenseCarbonPowder = std_solid('Dense Carbon Powder')
    StabilizedCarbon = std_solid('Stabilized Carbon')
    Xiranite = std_solid('Xiranite')
    HeavyXiranite = std_solid('Heavy Xiranite')
    Xircon = std_solid('Xircon')
    Hetonite = std_solid('Hetonite')
    CupriumBottle = std_solid('Cuprium Bottle')
    FerriumBottle = std_solid('Ferrium Bottle')
    AmethystBottle = std_solid('Amethyst Bottle')
    SteelBottle = std_solid('Steel Bottle')
    CrystonBottle = std_solid('Cryston Bottle')
    HetoniteBottle = std_solid('Hetonite Bottle')
    CupriumPart = std_solid('Cuprium Part')
    FerriumPart = std_solid('Ferrium Part')
    AmethystPart = std_solid('Amethyst Part')
    SteelPart = std_solid('Steel Part')
    CrystonPart = std_solid('Cryston Part')
    HetonitePart = std_solid('Hetonite Part')
    CupriumBottlefilledwithYazhenSolution = std_solid('Cuprium Bottle filled with Yazhen Solution')
    CupriumBottlefilledwithJincaoSolution = std_solid('Cuprium Bottle filled with Jincao Solution')
    YazhenSyringeA = std_solid('Yazhen Syringe A')
    JincaoTea = std_solid('Jincao Tea')
    IndustrialExplosive = std_solid('Industrial Explosive')
    LCValleyBattery = std_solid('LC Valley Battery')
    SCValleyBattery = std_solid('SC Valley Battery')
    HCValleyBattery = std_solid('HC Valley Battery')
    FerriumBottlefilledwithYazhenSolution = std_solid('Ferrium Bottle filled with Yazhen Solution')
    YazhenSyringeC = std_solid('Yazhen Syringe C')
    FerriumBottlefilledwithJincaoSolution = std_solid('Ferrium Bottle filled with Jincao Solution')
    JincaoDrink = std_solid('Jincao Drink')
    LCWulingBattery = std_solid('LC Wuling Battery')
    SCWulingBattery = std_solid('SC Wuling Battery')
    GroundBuckflowerPowder = std_solid('Ground Buckflower Powder')
    GroundCitromePowder = std_solid('Ground Citrome Powder')
    FerriumComponent = std_solid('Ferrium Component')
    CrystonComponent = std_solid('Cryston Component')
    AmethystComponent = std_solid('Amethyst Component')
    XiraniteComponent = std_solid('Xiranite Component')
    CupriumComponent = std_solid('Cuprium Component')
    HetoniteComponent = std_solid('Hetonite Component')
    Pyrrolite = std_solid('Pyrrolite')
    PyrrolitePart = std_solid('Pyrrolite Part')
    CupriumCanister = std_solid('Cuprium Canister')
    PyrroliteComponent = std_solid('Pyrrolite Component')
    SeparatorCore = std_solid('Separator Core')

    Water = std_liquid('Water')
    Acid = std_liquid('Acid')
    Sewage = std_liquid('Sewage')
    YazhenSolution = std_liquid('Yazhen Solution')
    JincaoSolution = std_liquid('Jincao Solution')
    LiquidXiranite = std_liquid('Liquid Xiranite')
    LiquidHeavyXiranite = std_liquid('Liquid Heavy Xiranite')
    CupriumSolution = std_liquid('Cuprium Solution')
    XirconEffluent = std_liquid('Xircon Effluent')
    InertXirconEffluent = std_liquid('Inert Xircon Effluent')
    HetoniteSolution = std_liquid('Hetonite Solution')

    Inergen = std_gas('Inergen')
    Aquagen = std_gas('Aquagen')
    Acridgen = std_gas('Acridgen')
    Xiragen = std_gas('Xiragen')
    HeavyXiragen = std_gas('Heavy Xiragen')
    CupriumGas = std_gas('Cuprium Gas')
    HetoniteGas = std_gas('Hetonite Gas')
    PyrroliteGas = std_gas('Pyrrolite Gas')

    StableENV = Material(name='Stable ENV', tags=VIRTUAL)
    HumidENV = Material(name='Humid ENV', tags=VIRTUAL)
    AcridENV = Material(name='Acrid ENV', tags=VIRTUAL)
    XiraniteENV = Material(name='Xiranite ENV', tags=VIRTUAL)