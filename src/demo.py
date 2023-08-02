import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from training_functions import smooth_data
from lstm import lstm_model
from training_functions import get_data_and_model
from utils import change_to_working_directory

def demo_data(input_data_ppg, input_data_ecg, ground_truth_data, output_data, demo_path="", frames=500, interval=20, x_range=500):
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches(18, 9)

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    xdata1, ydata1, xdata2, ydata2, xdata3, ydata3, xdata4, ydata4 = [], [], [], [], [], [], [], []
    ln1, = ax1.plot([], [], label='ppg input', color = "orange")
    ln2, = ax2.plot([], [], label='ecg input', color = "orange")
    ln3, = ax3.plot([], [], label='abp ground truth', color = "orange")
    ln4, = ax3.plot([], [], label='abp prediction', color = "blue")

    def init():
        ax1.set_xlim(0, x_range)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Scaled PPG')
        ax1.set_title('PPG Input')
        ax1.legend()

        ax2.set_xlim(0, x_range)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Scaled ECG')
        ax2.set_title('ECG Input')
        ax2.legend()

        ax3.set_xlim(0, x_range)
        ax3.set_ylim(0, 0.5)
        ax3.set_ylabel('Scaled ABP')
        ax3.set_title('ABP Test Results')
        ax3.legend()

        return ln1, ln2, ln3, ln4,
    
    def update(frame):
        xdata1.append(frame)
        ydata1.append(input_data_ppg[frame])
        ln1.set_data(xdata1, ydata1)

        xdata2.append(frame)
        ydata2.append(input_data_ecg[frame])
        ln2.set_data(xdata2, ydata2)

        xdata3.append(frame)
        ydata3.append(ground_truth_data[frame])
        ln3.set_data(xdata3, ydata3)
        
        xdata4.append(frame)
        ydata4.append(output_data[frame])
        ln4.set_data(xdata4, ydata4)
        
        if frame >= 3 * x_range / 4:
            ax1.set_xlim(frame - 3 * x_range / 4, frame + x_range / 4)
            ax2.set_xlim(frame - 3 * x_range / 4, frame + x_range / 4)
            ax3.set_xlim(frame - 3 * x_range / 4, frame + x_range / 4)
        
        return ln1, ln2, ln3, ln4
    
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=10, init_func=init, blit=False, repeat=False)

    if demo_path == "":
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30)
        anim.save(demo_path, writer=writergif)

    plt.close()

def plot_all(input_data_ppg, input_data_ecg, ground_truth_data, output_data):
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches(18, 9)

    ax1 = fig.add_subplot(311)
    ax1.plot(input_data_ppg, label='input', color="orange")
    ax1.set_ylabel('scaled PPG')
    ax1.set_title('PPG Input')
    ax1.legend()

    ax2 = fig.add_subplot(311)
    ax2.plot(input_data_ecg, label='input', color="orange")
    ax2.set_ylabel('scaled ECG')
    ax2.set_title('ECG Input')
    ax2.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(ground_truth_data, label='ground truth', color="orange")
    ax3.plot(output_data, label='prediction', color="blue")
    ax3.set_ylabel('scaled ABP')
    ax3.set_title('Test Results')
    ax3.legend()

    plt.show()

def main():
    change_to_working_directory()

    model, _, _, _, encoder_input_test, decoder_output_test = get_data_and_model(input_dim=2, hidden_dim=128, layer_dim=3, output_dim=1)
    state = torch.load("./models/model_1.pt")
    model.load_state_dict(state)
    print(model.eval())

    input_data_ppg, input_data_ecg, ground_truth_data, output_data = smooth_data(model, encoder_input_test, decoder_output_test)
    demo_data(input_data_ppg, input_data_ecg, ground_truth_data, output_data, demo_path='./demos/demo.gif', frames=len(output_data), interval=1, x_range=400)
    #demo_data(input_data_ppg, input_data_ecg, ground_truth_data, output_data, frames=len(output_data), interval=1, x_range=400)

if __name__ == "__main__":
    main()