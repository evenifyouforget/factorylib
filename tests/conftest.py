test_data = []

for i in range(11):
    a = i / 10
    for j in range(11):
        b = j / 10
        test_data.append((a, b))

def pytest_generate_tests(metafunc):
    if 'in_vec2' in metafunc.fixturenames:
        metafunc.parametrize('in_vec2', test_data)
