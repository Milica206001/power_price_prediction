from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function returns the list of requirements
    '''
    requirements=[]
    with open('requirements.txt') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

setup(
    name='power_price_prediction',
    version='0.0.1',
    author='Milica'
    author_email='mmilica206@gmail.com',
    packages=find_packages()
    install_requires=get_requirements('requirements.txt')

)