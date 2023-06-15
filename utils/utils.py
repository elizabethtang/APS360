import os

def change_to_working_directory():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir+"/../..")