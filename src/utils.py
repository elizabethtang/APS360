import os
import matplotlib.pyplot as plt

def change_to_working_directory():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir+"/..")

def plot_loss(loss_curve):
    plt.plot(loss_curve)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('training curve')