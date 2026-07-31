import argparse
import graphviz
import numpy as np
from numpy import inf
from scipy.optimize import Bounds, LinearConstraint, milp
from fractions import Fraction
import textwrap

_unique_counter = 0

SOLID = 'S'
LIQUID = 'L'
GAS = 'G'
VIRTUAL = 'V'
HIDDEN = 'H'
_EPS = 1e-12
FRACTION_PREMULTIPLY = 12
FRACTION_LIMIT_DENOM = 8
DECREMENT = 1 / FRACTION_PREMULTIPLY / FRACTION_LIMIT_DENOM ** 2
LABEL_WIDTH = 40

class Fraction2(Fraction):
    """
    Fraction with customized printing
    """
    def __str__(self):
        as_fraction = Fraction.__str__(self)
        as_float = float(self)
        return f'[{as_fraction} = {as_float}]'

def find_close_fraction(x, force_fractions=False, allow_greater=False, allow_negative=False):
    """
    force_fractions=False mode: try to find a close fraction, or else return the original value.
    force_fractions=True mode: always returns a fraction, subject to constraints.
    """
    def round_to_fraction(x):
        return Fraction2(Fraction2(x * FRACTION_PREMULTIPLY).limit_denominator(FRACTION_LIMIT_DENOM) / FRACTION_PREMULTIPLY)
    x_as_frac = round_to_fraction(x)
    if not force_fractions:
        if np.isclose(x_as_frac, x, rtol=_EPS, atol=_EPS):
            return x_as_frac
        return x
    x_modified = x
    while not allow_greater and x_as_frac >= x + _EPS:
        x_modified -= DECREMENT
        if not allow_negative and x_modified < 0:
            return Fraction2(0)
        x_as_frac = round_to_fraction(x_modified)
    return x_as_frac
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
        if not other:
            return self
        return AddMaterial(self, other).simplify()
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + other * -1
    def __rsub__(self, other):
        return other + self * -1
    def __mul__(self, other):
        if other == 1:
            return self
        return MulMaterial(self, other).simplify()
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * -1
    def gather_materials(self):
        return {self}
    def __str__(self):
        return f'{self.name}'
    def simplify(self):
        return self
    def is_negative(self):
        return False
    def split(self):
        return (0, self)

class AddMaterial(Material):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def substitute(self, subs_dict):
        return substitute(self.lhs, subs_dict) + substitute(self.rhs, subs_dict)
    def gather_materials(self):
        return gather_materials(self.lhs) | gather_materials(self.rhs)
    def __str__(self):
        if self.rhs.is_negative():
            return f'{self.lhs} - {-1 * self.rhs}'
        return f'{self.lhs} + {self.rhs}'
    def simplify(self):
        # try to expand sum
        queue = [self.lhs, self.rhs]
        others = []
        while queue:
            x = queue.pop()
            if isinstance(x, AddMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        result = others[0]
        for x in others[1:]:
            result = AddMaterial(result, x)
        return result
    def split(self):
        queue = [self.lhs, self.rhs]
        others = []
        while queue:
            x = queue.pop()
            if isinstance(x, AddMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        pos = []
        neg = []
        for x in others:
            if x.is_negative():
                neg.append(-x)
            else:
                pos.append(x)
        return sum(neg), sum(pos)

class MulMaterial(Material):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def substitute(self, subs_dict):
        return substitute(self.lhs, subs_dict) * substitute(self.rhs, subs_dict)
    def gather_materials(self):
        return gather_materials(self.lhs) | gather_materials(self.rhs)
    def __str__(self):
        lhs = self.lhs
        rhs = self.rhs
        if isinstance(rhs, (int, float)) and isinstance(lhs, Material):
            lhs, rhs = rhs, lhs
        if isinstance(lhs, (int, float)) and type(rhs) is Material:
            return f'{lhs}{rhs.unit} {rhs.name}'
        if lhs == -1:
            return f'-{rhs}'
        return f'{lhs}×{rhs}'
    def simplify(self):
        # try to expand product
        queue = [self.lhs, self.rhs]
        constant = 1
        others = []
        while queue:
            x = queue.pop()
            if isinstance(x, (int, float)):
                constant *= x
            elif isinstance(x, MulMaterial):
                queue.append(x.lhs)
                queue.append(x.rhs)
            else:
                others.append(x)
        others = others[::-1]
        result = others[0]
        for x in others[1:]:
            result = MulMaterial(result, x)
        result = MulMaterial(constant, result)
        return result
    def is_negative(self):
        lhs = self.lhs
        rhs = self.rhs
        if isinstance(rhs, (int, float)) and isinstance(lhs, Material):
            lhs, rhs = rhs, lhs
        if isinstance(lhs, (int, float)) and type(rhs) is Material:
            return lhs < 0
        return False
            

def substitute(expr, subs_dict):
    if isinstance(expr, (int, float)):
        return expr
    return expr.substitute(subs_dict)


class Recipe(object):
    def __init__(self, expression, name, max_multiples=inf, integer_only=False):
        self.expression = expression.simplify()
        self.name = name
        self.max_multiples = max_multiples
        self.integer_only = integer_only
    def gather_materials(self):
        return gather_materials(self.expression)
    def nice_expression_str(self):
        neg, pos = self.expression.split()
        return f'{neg} --> {pos}'


def gather_materials(expr):
    if isinstance(expr, list):
        result = set()
        for ex in expr:
            result |= gather_materials(ex)
        return result
    if isinstance(expr, (int, float)):
        return set()
    return expr.gather_materials()

def wrap_label(text):
    return "\n".join([textwrap.fill(line, width=LABEL_WIDTH) for line in text.splitlines()])

def optimize(all_materials, all_recipes, material_to_maximize, force_fractions=False, graph_outfile=None):
    # Get all recipes
    num_recipes = len(all_recipes)
    # Get the complete list of all materials, including recipe counters
    all_materials = sorted(set(all_materials) | gather_materials(all_recipes))
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
    all_recipes_multiples = res.x
    all_recipes_multiples = np.array([find_close_fraction(x, force_fractions=force_fractions) for x in all_recipes_multiples], dtype=object)
    for i, multiples in enumerate(all_recipes_multiples):
        if multiples == 0:
            continue
        print(f'- {multiples} multiples of {all_recipes[i].name}')
    print('## Balance Sheet Per Material')
    plus_amount = np.maximum(0, recipe_matrix.T) @ all_recipes_multiples
    net_amount = recipe_matrix.T @ all_recipes_multiples
    dot = None
    node_names = {}
    hidden_node_mats = set()
    if graph_outfile:
        dot = graphviz.Digraph(engine='sfdp', graph_attr={'overlap_scaling': '-10'})
        for i, material in enumerate(all_materials):
            node_names[material] = inode_name = f'material{i}'
            iplus = plus_amount[i]
            inet = net_amount[i]
            if iplus == 0 and inet == 0 or HIDDEN in material.tags:
                hidden_node_mats.add(material)
                continue
            isub = iplus - inet
            dot.node(inode_name, wrap_label(f'{material}\n\n+{iplus}{material.unit} - {isub}{material.unit}\n\n={inet}{material.unit}'))
    bits = [[f'### {material.name} (net {net}{material.unit})'] for material, net in zip(all_materials, net_amount)]
    for i, multiples in enumerate(all_recipes_multiples):
        if multiples == 0:
            continue
        recipe = all_recipes[i]
        nodef = None
        if dot:
            inode_name = f'recipe{i}'
            nodef = (inode_name, wrap_label(f'{recipe.name}\n\n{multiples} multiples'))
        edgefs = []
        for j, material in enumerate(all_materials):
            per_multiple = recipe_matrix[i,j]
            if per_multiple == 0:
                continue
            contribution = multiples * per_multiple
            bits[j].append(f'- {contribution}{material.unit} from {multiples} multiples of {recipe.name}')
            if dot and material not in hidden_node_mats:
                edge_in = inode_name
                edge_out = node_names[material]
                if contribution < 0:
                    edge_in, edge_out = edge_out, edge_in
                    contribution = -contribution
                edgefs.append((edge_in, edge_out, f'{contribution}{material.unit}'))
        if len(edgefs) >= 2:
            # only show recipes that have 2 or more connections
            dot.node(*nodef)
            for edgef in edgefs:
                dot.edge(*edgef)
    bits.sort()
    print('\n'.join(map('\n'.join, bits)))
    if dot:
        dot.render(graph_outfile)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-fractions', action='store_true', help='Forces all printed quantities to be an exact fraction, even if the original quantity may not be near any simple fraction. Results in an approximately satisfiable solution that is easier to build.')
    parser.add_argument('-t', '--target', choices=['sellable', 'mixed'], help='Which goal to optimize for')
    parser.add_argument('-o', '--graph-outfile', help='File to render graph to')
    args = parser.parse_args()
    PMIN = '/min'
    goal_material = WulingStockBill = Material(name='$', unit=PMIN, tags=VIRTUAL)
    Watt = Material(name='W', tags=VIRTUAL+HIDDEN)
    def std_alloc(name, additional_tags=''):
        return Material(name=f'Allocation: [{name}]', tags=VIRTUAL+additional_tags)
    ForgeAllocation = std_alloc('Forge of the Sky')
    MetatransferAllocation = std_alloc('Metatransfer')
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
    BuckflowerSeed = std_solid('Buckflower Seed')
    CitromeSeed = std_solid('Citrome Seed')
    AketineSeed = std_solid('Aketine Seed')
    SandleafSeed = std_solid('Sandleaf Seed')
    YazhenSeed = std_solid('Yazhen Seed')
    JincaoSeed = std_solid('Jincao Seed')
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
    Amethyst = std_solid('Amethyst Fiber')
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
    all_recipes = []
    #all_recipes.append(Recipe(expression=540 * OriginiumOre + 120 * FerriumOre + 420 * CupriumOre + 460 * Inergen + 100 * Xiragen + 12 * ForgeAllocation + 1 * MetatransferAllocation, name='Starting Materials', max_multiples=1))
    all_recipes.append(Recipe(expression=540 * OriginiumOre + 120 * FerriumOre + 420 * CupriumOre + 460 * Inergen + 12 * ForgeAllocation + 1 * MetatransferAllocation, name='Starting Materials', max_multiples=1))
    def std_building(building_name, power):
        def make_recipe(inputs, outputs, /, max_multiples=inf, integer_only=False, integer_inputs=None):
            power_str = f' ({power} W)' if power else ''
            name = f'{building_name}{power_str}: {inputs} --> {outputs}'
            additional_tags = '' if integer_inputs else HIDDEN
            counter = std_alloc(name, additional_tags=additional_tags)
            counter_inputs = power * Watt
            if integer_inputs:
                counter_inputs = counter_inputs + integer_inputs
            all_recipes.append(Recipe(-counter_inputs + counter, name=f'Assign Allocation: {name}', integer_only=True))
            all_recipes.append(Recipe(-counter - inputs + outputs, name=name, max_multiples=max_multiples, integer_only=integer_only))
        return make_recipe
    for metatransfer_option in [ # non-exhaustive list
        1500/60 * OriginiumOre,
        1500/60 * AmethystOre,
        1500/60 * DenseOriginiumPowder,
        1500/60 * FerriumOre,
        1500/60/2 * Steel,
        1500/60/50 * HCValleyBattery,
        1500/60/20 * SCValleyBattery,
        1500/60/20 * LCValleyBattery,
        ]:
        all_recipes.append(Recipe(-MetatransferAllocation + metatransfer_option, f'Choose Metatransfer: {metatransfer_option}', integer_only=True))
    std_pump = std_building('Fluid Pump', 10)
    std_pump(0, 60 * Water)
    std_pump2 = std_building('Acid Resistant Pump Mk II', 20)
    std_pump2(0, 60 * Acid)
    std_refine = std_building('Refining Unit', 5)
    std_refine(30 * CupriumOre + 30 * Water, 30 * Cuprium + 30 * Sewage)
    std_refine(30 * FerriumOre, 30 * Ferrium)
    std_refine(30 * FerriumPowder, 30 * Ferrium)
    std_refine(30 * AmethystOre, 30 * Amethyst)
    std_refine(30 * OriginiumOre, 30 * Origocrust)
    std_refine(30 * DenseOrigocrustPowder, 30 * PackedOrigocrust)
    std_refine(30 * DenseFerriumPowder, 30 * Steel)
    std_refine(30 * CrystonPowder, 30 * CrystonFiber)
    std_refine(30 * DenseCarbonPowder, 30 * StabilizedCarbon)
    std_refine(30 * DenseOriginiumPowder, 30 * DenseOrigocrustPowder)
    std_refine(30 * Buckflower, 30 * Carbon)
    std_refine(30 * Sandleaf, 30 * Carbon)
    #std_refine(30 * Jincao, 60 * Carbon)
    std_refine(30 * Yazhen, 60 * Carbon)
    std_shred = std_building('Shredding Unit', 5)
    std_shred(30 * Cuprium, 30 * CupriumPowder)
    std_shred(30 * Ferrium, 30 * FerriumPowder)
    std_shred(30 * Amethyst, 30 * AmethystPowder)
    std_shred(30 * OriginiumOre, 30 * OriginiumPowder)
    std_shred(30 * Carbon, 60 * CarbonPowder)
    std_shred(30 * Origocrust, 30 * OrigocrustPowder)
    std_shred(30 * Buckflower, 60 * BuckflowerPowder)
    std_shred(30 * Citrome, 60 * CitromePowder)
    std_shred(30 * Sandleaf, 90 * SandleafPowder)
    std_shred(30 * Aketine, 60 * AketinePowder)
    std_shred(30 * Jincao, 60 * JincaoPowder)
    std_shred(30 * Yazhen, 60 * YazhenPowder)
    std_fit = std_building('Fitting Unit', 20)
    std_fit(30 * Ferrium, 30 * FerriumPart)
    std_fit(30 * Amethyst, 30 * AmethystPart)
    std_fit(30 * Steel, 30 * SteelPart)
    std_fit(30 * CrystonFiber, 30 * CrystonPart)
    std_fit(30 * Cuprium, 30 * CupriumPart)
    std_fit(30 * Hetonite, 6 * HetonitePart)
    std_fit(30 * Pyrrolite, 6 * PyrrolitePart)
    std_mould = std_building('Moulding Unit', 10)
    std_mould(60 * Cuprium + 30 * Inergen, 30 * CupriumCanister)
    std_mould(60 * Ferrium, 30 * FerriumBottle)
    std_mould(60 * Amethyst, 30 * AmethystBottle)
    std_mould(60 * Steel, 30 * SteelBottle)
    std_mould(60 * CrystonFiber, 30 * CrystonBottle)
    std_mould(60 * Cuprium, 30 * CupriumBottle)
    std_mould(60 * Hetonite, 30 * HetoniteBottle)
    std_plant = std_building('Planting Unit', 20)
    std_plant(30 * JincaoSeed + 30 * Water, 60 * Jincao)
    std_plant(30 * YazhenSeed + 30 * Water, 60 * Yazhen)
    std_plant(30 * BuckflowerSeed, 30 * Buckflower)
    std_plant(30 * CitromeSeed, 30 * Citrome)
    std_plant(30 * SandleafSeed, 30 * Sandleaf)
    std_plant(30 * AketineSeed, 30 * Aketine)
    std_seedpick = std_building('Seed-Picking Unit', 10)
    std_seedpick(30 * Buckflower, 60 * BuckflowerSeed)
    std_seedpick(30 * Citrome, 60 * CitromeSeed)
    std_seedpick(30 * Sandleaf, 60 * SandleafSeed)
    std_seedpick(30 * Aketine, 60 * AketineSeed)
    std_seedpick(30 * Jincao, 30 * JincaoSeed)
    std_seedpick(30 * Yazhen, 30 * YazhenSeed)
    std_treatment = std_building('Water Treatment Unit', 50)
    std_treatment(30 * Sewage, 0)
    std_treatment(30 * XirconEffluent, 0)
    std_treatment(30 * InertXirconEffluent, 0)
    std_gear = std_building('Gearing Unit', 10)
    std_gear(30 * Origocrust + 30 * Amethyst, 6 * AmethystComponent)
    std_gear(60 * Origocrust + 60 * Ferrium, 6 * FerriumComponent)
    std_gear(60 * PackedOrigocrust + 60 * CrystonFiber, 6 * CrystonComponent)
    std_gear(60 * PackedOrigocrust + 60 * Xiranite, 6 * XiraniteComponent)
    std_gear(60 * CupriumPart + 60 * Xiranite, 6 * CupriumComponent)
    std_gear(12 * HetonitePart + 12 * HeavyXiranite, 6 * HetoniteComponent)
    std_gear(6 * PyrrolitePart + 12 * HeavyXiranite, 6 * PyrroliteComponent)
    std_fill = std_building('Filling Unit', 20)
    std_fill(30 * CupriumBottle + 30 * YazhenSolution, 30 * CupriumBottlefilledwithYazhenSolution)
    std_fill(30 * CupriumBottle + 30 * JincaoSolution, 30 * CupriumBottlefilledwithJincaoSolution)
    std_fill(30 * FerriumBottle + 30 * YazhenSolution, 30 * FerriumBottlefilledwithYazhenSolution)
    std_fill(30 * FerriumBottle + 30 * JincaoSolution, 30 * FerriumBottlefilledwithJincaoSolution)
    std_pack = std_building('Packaging Unit', 20)
    std_pack(30 * AmethystPart + 6 * AketinePowder, 6 * IndustrialExplosive)
    std_pack(30 * AmethystPart + 60 * OriginiumPowder, 6 * LCValleyBattery)
    std_pack(60 * FerriumPart + 90 * OriginiumPowder, 6 * SCValleyBattery)
    std_pack(60 * SteelPart + 90 * DenseOriginiumPowder, 6 * HCValleyBattery)
    std_pack(60 * FerriumPart + 30 * FerriumBottlefilledwithYazhenSolution, 6 * YazhenSyringeC)
    std_pack(60 * FerriumPart + 30 * FerriumBottlefilledwithJincaoSolution, 6 * JincaoDrink)
    std_pack(60 * CupriumPart + 30 * CupriumBottlefilledwithYazhenSolution, 6 * YazhenSyringeA)
    std_pack(60 * CupriumPart + 30 * CupriumBottlefilledwithJincaoSolution, 6 * JincaoTea)
    std_pack(30 * Xiranite + 90 * DenseOriginiumPowder, 6 * LCWulingBattery)
    std_pack(30 * Xircon + 120 * DenseOriginiumPowder, 6 * SCWulingBattery)
    std_pack(30 * CupriumCanister + 30 * Xiranite, 60 * SeparatorCore)
    std_grind = std_building('Grinding Unit', 50)
    std_grind(60 * FerriumPowder + 30 * SandleafPowder, 30 * DenseFerriumPowder)
    std_grind(60 * AmethystPowder + 30 * SandleafPowder, 30 * CrystonPowder)
    std_grind(60 * OriginiumPowder + 30 * SandleafPowder, 30 * DenseOriginiumPowder)
    std_grind(60 * CarbonPowder + 30 * SandleafPowder, 30 * DenseCarbonPowder)
    std_grind(60 * OrigocrustPowder + 30 * SandleafPowder, 30 * DenseOrigocrustPowder)
    std_grind(60 * BuckflowerPowder + 30 * SandleafPowder, 30 * GroundBuckflowerPowder)
    std_grind(60 * CitromePowder + 30 * SandleafPowder, 30 * GroundCitromePowder)
    std_reactor = std_building('Reactor Crucible', 50)
    std_reactor(30 * JincaoPowder + 30 * Water, 30 * JincaoSolution)
    std_reactor(30 * YazhenPowder + 30 * Water, 30 * YazhenSolution)
    std_reactor(30 * Xiranite + 30 * Water, 30 * LiquidXiranite)
    std_reactor(30 * HeavyXiranite + 30 * Acid, 30 * LiquidHeavyXiranite)
    std_reactor(30 * CupriumPowder + 30 * Acid, 30 * CupriumSolution)
    std_reactor(30 * LiquidXiranite + 30 * Sewage, 30 * XirconEffluent + 30 * InertXirconEffluent)
    std_reactor(60 * XirconEffluent + 30 * FerriumPowder, 30 * Xircon + 30 * Sewage)
    std_reactor(60 * HetoniteSolution + 30 * FerriumPowder, 30 * Hetonite + 30 * Sewage)
    std_forge = std_building('Forge of the Sky', 50)
    std_forge(60 * StabilizedCarbon + 30 * Water, 30 * Xiranite, integer_inputs=ForgeAllocation)
    std_forge(60 * Xiranite + 30 * XirconEffluent, 6 * HeavyXiranite, integer_inputs=ForgeAllocation)
    std_forge(30 * Carbon + 30 * Water, 30 * Xiranite, integer_inputs=ForgeAllocation + StableENV)
    std_purify = std_building('Purification Unit', 50)
    std_purify(60 * Xiragen + 60 * SeparatorCore, 30 * HeavyXiragen)
    std_purify(60 * Xiragen + 30 * SeparatorCore, 30 * HeavyXiragen, integer_inputs=StableENV)
    std_purify(60 * CupriumGas + 60 * SeparatorCore, 30 * HetoniteGas)
    std_purify(60 * CupriumGas + 30 * SeparatorCore, 30 * HetoniteGas, integer_inputs=StableENV)
    std_purify(120 * InertXirconEffluent, 30 * XirconEffluent + 30 * Water)
    std_purify(120 * CupriumSolution, 30 * HetoniteSolution + 30 * Acid)
    std_lg_transmute = std_building('Fluid-Gas Transmuting Unit', 50)
    def std_lg_transmute_pair(lside, gside):
        std_lg_transmute(lside, gside, integer_inputs=6*LiquidXiranite)
        std_lg_transmute(gside, lside, integer_inputs=6*LiquidXiranite)
    std_lg_transmute_pair(30 * Water, 30 * Aquagen)
    std_lg_transmute_pair(30 * Acid, 30 * Acridgen)
    std_lg_transmute_pair(30 * LiquidXiranite, 30 * Xiragen)
    std_lg_transmute_pair(12 * LiquidHeavyXiranite, 30 * HeavyXiragen)
    std_lg_transmute_pair(60 * CupriumSolution, 30 * CupriumGas)
    std_lg_transmute_pair(30 * HetoniteSolution, 30 * HetoniteGas)
    std_sg_transmute = std_building('Solid-Gas Transmuting Unit', 50)
    def std_sg_transmute_pair(sside, gside):
        std_sg_transmute(sside, gside, integer_inputs=6*Xiragen)
        std_sg_transmute(gside, sside, integer_inputs=6*Xiragen)
    std_sg_transmute_pair(30 * Xiranite, 30 * Xiragen)
    std_sg_transmute_pair(12 * HeavyXiranite, 30 * HeavyXiragen)
    std_sg_transmute_pair(60 * Cuprium, 30 * CupriumGas)
    std_sg_transmute_pair(30 * Hetonite, 60 * HetoniteGas)
    std_sg_transmute_pair(30 * Pyrrolite, 30 * PyrroliteGas)
    std_gas_reactor = std_building('Gas Reactor Globe', 50)
    std_gas_reactor(60 * HetoniteGas + 30 * Xiragen, 30 * PyrroliteGas, integer_inputs=AcridENV)
    std_field = std_building('Gas Dispersing Unit', 0.01) # real cost is 0, but we don't want LP to treat it as free
    std_field(6 * Inergen, 4 * StableENV, integer_only=True)
    std_field(6 * Aquagen, 4 * HumidENV, integer_only=True)
    std_field(6 * Acridgen, 4 * AcridENV, integer_only=True)
    std_field(6 * Xiragen, 4 * XiraniteENV, integer_only=True)
    std_thermal = std_building('Thermal Bank', 0)
    std_thermal(7.5 * OriginiumOre, 50 * Watt)
    std_thermal(1.5 * LCValleyBattery, 220 * Watt)
    std_thermal(1.5 * SCValleyBattery, 420 * Watt)
    std_thermal(1.5 * HCValleyBattery, 1100 * Watt)
    std_thermal(1.5 * LCWulingBattery, 1600 * Watt)
    std_thermal(1.5 * SCWulingBattery, 3200 * Watt)
    std_sell = std_building('Sell', 0)
    std_sell(Xiranite, WulingStockBill)
    std_sell(CupriumPart, WulingStockBill)
    std_sell(SeparatorCore, WulingStockBill)
    std_sell(YazhenSyringeC, 16 * WulingStockBill)
    #std_sell(JincaoDrink, 16 * WulingStockBill)
    std_sell(YazhenSyringeA, 22 * WulingStockBill)
    #std_sell(JincaoTea, 22 * WulingStockBill)
    std_sell(LCWulingBattery, 25 * WulingStockBill)
    std_sell(HeavyXiranite, 27 * WulingStockBill)
    std_sell(HetonitePart, 48 * WulingStockBill)
    std_sell(SCWulingBattery, 54 * WulingStockBill)
    std_sell(PyrrolitePart, 70 * WulingStockBill)
    std_test_area = std_building('Test Area Purification Node', 0)
    #std_test_area(30 * Sewage, XirconEffluent, max_multiples=12)
    if args.target == 'mixed':
        goal_material = PerformancePoint = Material(name='pp', tags=VIRTUAL+HIDDEN)
        def pp_satisfaction(target_mat, mat_amount, pp_worth):
            target_expr = mat_amount * target_mat
            First100Percent = std_alloc(f'First 100% of {target_expr}', additional_tags=HIDDEN)
            std_pt = std_building(f'Award Points For {target_mat}', 0)
            std_pt(target_expr, First100Percent, max_multiples=1)
            std_pt(First100Percent, pp_worth * 0.1 * PerformancePoint)
            std_pt(First100Percent, pp_worth * 0.8 * PerformancePoint, integer_only=True)
            # N segments with same output but different input
            # smallest input one is the best and will be taken first
            N_SEGMENTS = 30
            R = 0.85
            for i in range(N_SEGMENTS):
                std_pt(mat_amount * R ** i * target_mat, pp_worth * 0.2 / N_SEGMENTS * PerformancePoint, max_multiples=1)
        pp_satisfaction(WulingStockBill, 1600, 10000)
        # since the factory already covers its own power cost, the power goal is only for additional buildings
        pp_satisfaction(Watt, 2500, 10000)
        def pp_nonzero(target_mat, mat_amount, pp_worth):
            std_pt = std_building(f'Award Points For {target_mat}', 0)
            # N segments with same output but different input
            # smallest input one is the best and will be taken first
            N_SEGMENTS = 100
            R = 1.1
            in_amounts = [R ** i for i in range(N_SEGMENTS)]
            mul = mat_amount / sum(in_amounts[:N_SEGMENTS//2])
            for i in range(N_SEGMENTS):
                std_pt(mul * in_amounts[i] * target_mat, pp_worth / N_SEGMENTS * PerformancePoint, max_multiples=1)
        CraftingPoint = []
        for i in range(4):
            CraftingPoint.append(Material(name=f'Wuling Tier {i+1} Gear', unit=PMIN, tags=VIRTUAL))
        std_craft = std_building('Gear Crafting', 0)
        std_craft(50 * XiraniteComponent, CraftingPoint[0])
        std_craft(50 * CupriumComponent, CraftingPoint[1])
        std_craft(50 * HetoniteComponent, CraftingPoint[2])
        std_craft(50 * PyrroliteComponent, CraftingPoint[3])
        std_craft(CraftingPoint[1], CraftingPoint[0])
        std_craft(CraftingPoint[2], 5 * CraftingPoint[1])
        std_craft(CraftingPoint[3], 2 * CraftingPoint[2])
        pp_nonzero(CraftingPoint[0], 0.05, 100)
        pp_nonzero(CraftingPoint[1], 0.05, 100)
        pp_nonzero(CraftingPoint[2], 0.05, 100)
        pp_nonzero(CraftingPoint[3], 0.1, 200)
        # materials that already occur in other pipelines can be siphoned to a stash in storage mode
        # or for liquids, a fluid tank
        # after quickly saving up the small amount, it will not siphon anymore from main production
        #pp_nonzero(CupriumCanister, 1, 200)
        #pp_nonzero(CupriumPart, 0.5, 100)
        pp_nonzero(HetonitePart, 0.5, 100)
        pp_nonzero(PyrrolitePart, 1, 500)
        #pp_nonzero(LiquidXiranite, 15, 200)
        #pp_nonzero(LiquidHeavyXiranite, 0.5, 100)
        # no need to model delivery jobs
        # 14000/day * 2 ~= 19.4/min
        # this is easily met with the excess of sellable goods (which we can't sell anyway)
        # or some unused starting solids
        # or with a single planting loop (30/min)
        # or sandleaf + sandleaf powder + carbon + carbon powder gives you 4 different solids
    optimize(set(), all_recipes, goal_material, force_fractions=args.force_fractions, graph_outfile=args.graph_outfile)
