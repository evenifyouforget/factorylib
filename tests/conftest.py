in2_tuples = []

for i in range(11):
    a = i / 10
    for j in range(11):
        b = j / 10
        in2_tuples.append((a, b))


def pytest_generate_tests(metafunc):
    if "in_vec2" in metafunc.fixturenames:
        metafunc.parametrize("in_vec2", in2_tuples)
