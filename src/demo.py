import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from training_functions import smooth_data
from lstm import lstm_model
from training_functions import get_data_and_model
from utils import change_to_working_directory

def demo_data(input_data, ground_truth_data, output_data, demo_path="", frames=500, interval=20, x_range=500):
    fig = plt.figure()
    fig.set_size_inches(18, 8)

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    xdata1, ydata1, xdata2, ydata2, xdata3, ydata3 = [], [], [], [], [], []
    ln1, = ax1.plot([], [], label='input', color = "orange")
    ln2, = ax2.plot([], [], label='ground truth', color = "orange")
    ln3, = ax2.plot([], [], label='prediction', color = "blue")

    def init():
        ax1.set_xlim(0, x_range)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('scaled PPG')
        ax1.set_title('PPG Input')
        ax1.legend()

        ax2.set_xlim(0, x_range)
        ax2.set_ylim(0, 0.5)
        ax2.set_ylabel('scaled ABP')
        ax2.set_title('Test Results')
        ax2.legend()

        return ln1, ln2, ln3,
    
    def update(frame):
        xdata1.append(frame)
        ydata1.append(input_data[frame])
        ln1.set_data(xdata1, ydata1)

        xdata2.append(frame)
        ydata2.append(ground_truth_data[frame])
        ln2.set_data(xdata2, ydata2)
        
        xdata3.append(frame)
        ydata3.append(output_data[frame])
        ln3.set_data(xdata3, ydata3)
        
        if frame >= 3 * x_range / 4:
            ax1.set_xlim(frame - 3 * x_range / 4, frame + x_range / 4)
            ax2.set_xlim(frame - 3 * x_range / 4, frame + x_range / 4)
        
        return ln1, ln2, ln3
    
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, init_func=init, blit=False, repeat=False)

    if demo_path == "":
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30)
        anim.save(demo_path, writer=writergif)

    plt.close()

def plot_all(input_data, ground_truth_data, output_data):
    fig = plt.figure()
    fig.set_size_inches(18, 8)

    ax1 = fig.add_subplot(211)
    ax1.plot(input_data, label='input', color="orange")
    ax1.set_ylabel('scaled PPG')
    ax1.set_title('PPG Input')
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.plot(ground_truth_data, label='ground truth', color="orange")
    ax2.plot(output_data, label='prediction', color="blue")
    ax2.set_ylabel('scaled ABP')
    ax2.set_title('Test Results')
    ax2.legend()

    plt.show()

def main():
    change_to_working_directory()

    model, train_loader, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test = get_data_and_model(input_dim=2, hidden_dim=128, layer_dim=3, output_dim=1)
    state = torch.load("./models/model_1.pt")
    model.load_state_dict(state)
    print(model.eval())

    input_data, ground_truth_data, output_data = smooth_data(model, encoder_input_test, decoder_output_test)
    demo_data(input_data, ground_truth_data, output_data, demo_path='./demos/demo_short.gif', frames=len(input_data), interval=1, x_range=400)
    #demo_data(input_data, ground_truth_data, output_data, frames=len(input_data), interval=1, x_range=400)

if __name__ == "__main__":
    main()