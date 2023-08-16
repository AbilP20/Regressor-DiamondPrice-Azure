from setuptools import find_packages,setup

IGNORE_E_DOT = '-e .'

def get_requirements(file_path : str)->list[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
    requirements = [i.replace('\n','') for i in requirements]
    if IGNORE_E_DOT in requirements:
        requirements.remove(IGNORE_E_DOT)


setup(
    name = 'Regressor Project',
    version = '0.0.1',
    author = 'Abil',
    author_email='abilpt@gmail.com',
    install_requires = ['numpy', 'pandas'],
    packages = find_packages()
)