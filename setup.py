from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    get requirements
    '''
    requirements=[]
    with open(file_path) as fObj:
        requirements=fObj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='model-deploy',
    version='0.0.1',
    author='Ryan Tri',
    author_email='ry4ntr1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
