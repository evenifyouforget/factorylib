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
        if not other:
            return self
        return AddMaterial(self, other)
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + other * -1
    def __rsub__(self, other):
        return other + self * -1
    def __mul__(self, other):
        if other == 1:
            return self
        return MulMaterial(self, other)
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * -1
    def gather_materials(self):
        return {self}
    def __str__(self):
        return f'{self.name}'

class AddMaterial(Material):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def substitute(self, subs_dict):
        return substitute(self.lhs, subs_dict) + substitute(self.rhs, subs_dict)
    def gather_materials(self):
        return gather_materials(self.lhs) | gather_materials(self.rhs)
    def __str__(self):
        return f'{self.lhs} + {self.rhs}'
    

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
        if isinstance(lhs, (int, float)) and isinstance(rhs, Material):
            return f'{lhs}{rhs.unit} {rhs.name}'
        return f'{lhs}{rhs}'

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
    def gather_materials(self):
        return gather_materials(self.expression)


def gather_materials(expr):
    if isinstance(expr, list):
        result = set()
        for ex in expr:
            result |= gather_materials(ex)
        return result
    if isinstance(expr, (int, float)):
        return set()
    return expr.gather_materials()

def optimize(all_materials, all_recipes, material_to_maximize):
    # Get all recipes
    num_recipes = len(all_recipes)
    # Get the complete list of all materials, including recipe counters
    all_materials = list(set(all_materials) | gather_materials(all_recipes))
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
    def std_alloc(name):
        return Material(name=f'Allocation: {name}', tags=VIRTUAL)
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
    all_recipes.append(Recipe(expression=540 * OriginiumOre + 120 * FerriumOre + 420 * CupriumOre + 460 * Inergen + 100 * Xiragen + 12 * ForgeAllocation + 1 * MetatransferAllocation, name='Starting Materials', max_multiples=1))
    def std_building(building_name, power):
        def make_recipe(inputs, outputs, /, max_multiples=inf, integer_only=False, integer_inputs=None):
            name = f'{building_name} ({power} W): {inputs} --> {outputs}'
            counter = std_alloc(name)
            counter_inputs = power * Watt
            if integer_inputs:
                counter_inputs = counter_inputs + integer_inputs
            all_recipes.append(Recipe(-counter_inputs + counter, name=f'Assign Allocation: {name}', integer_only=True))
            all_recipes.append(Recipe(-counter - inputs + outputs, name=name, max_multiples=max_multiples, integer_only=integer_only))
        return make_recipe
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
    std_refine(30 * Jincao, 60 * Carbon)
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
    std_fill(30 * CupriumBottle + 30 * YazhenSolution, CupriumBottlefilledwithYazhenSolution)
    std_fill(30 * CupriumBottle + 30 * JincaoSolution, CupriumBottlefilledwithJincaoSolution)
    std_fill(30 * FerriumBottle + 30 * YazhenSolution, FerriumBottlefilledwithYazhenSolution)
    std_fill(30 * FerriumBottle + 30 * JincaoSolution, FerriumBottlefilledwithJincaoSolution)
    std_pack = std_building('Packaging Unit', 20)
    std_pack(30 * AmethystPart + 6 * AketinePowder, 6 * IndustrialExplosive)
    std_pack(30 * AmethystPart + 60 * OriginiumPowder, 6 * LCValleyBattery)
    std_pack(60 * FerriumPart + 90 * OriginiumPowder, SCValleyBattery)
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
    std_forge(60 * Xiranite + 30 * XirconEffluent, 30 * HeavyXiranite, integer_inputs=ForgeAllocation)
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
    std_field = std_building('Gas Dispersing Unit', 0)
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
    std_sell(SCWulingBattery, 54 * WulingStockBill)
    std_sell(PyrrolitePart, 70 * WulingStockBill)
    optimize(set(), all_recipes, WulingStockBill)
