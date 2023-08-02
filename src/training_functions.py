import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from math import factorial
import numpy as np
import matplotlib.animation as animation
from lstm import lstm_model, train_lstm
from data_processing import get_and_save_data, get_train_test_data, convert_to_1d
from globals import DATA

def get_data_and_model(input_dim, hidden_dim, layer_dim, output_dim, batch_size = 128, time_window = 32):
    train_data, encoder_input_train, decoder_output_train, test_data, encoder_input_test, decoder_output_test = get_train_test_data(dir_path=DATA, timewindow=time_window)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    model = lstm_model(input_dim, hidden_dim, layer_dim, output_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)



    return model, train_loader, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test

def view_train_test_result(model, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_trainset = model(torch.from_numpy(encoder_input_train[0:1280]).to(device).float())
    output_testset = model(torch.from_numpy(encoder_input_test[0:1280]).to(device).float())

    output_testset_1d = convert_to_1d(output_testset.cpu().detach().numpy())
    decoder_output_test_1d = convert_to_1d(decoder_output_test)
    output_trainset_1d = convert_to_1d(output_trainset.cpu().detach().numpy())
    decoder_output_train_1d = convert_to_1d(decoder_output_train)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.plot(output_testset_1d[:800], label='test')
    plt.title('Validation Result')
    plt.ylabel('scaled ABP')
    plt.plot(decoder_output_test_1d[:800], label='ground truth')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(output_trainset_1d[:800], label='train')
    plt.title('Training Result')
    plt.ylabel('scaled ABP')
    plt.plot(decoder_output_train_1d[:800], label='ground truth')
    plt.legend()

# Code credit: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def smooth_data(model, encoder_input_test, decoder_output_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_testset = model(torch.from_numpy(encoder_input_test[0:1280]).to(device).float())
    output_testset_1d = convert_to_1d(output_testset.cpu().detach().numpy())

    encoder_input_test_ppg_ecg = np.transpose(encoder_input_test, (2, 1, 0))

    encoder_input_test_ppg_1d = encoder_input_test_ppg_ecg[0][0][:len(output_testset_1d)]
    encoder_input_test_ecg_1d = encoder_input_test_ppg_ecg[1][0][:len(output_testset_1d)]
    decoder_output_test_1d = convert_to_1d(decoder_output_test)[:len(output_testset_1d)]
    smooth_output_testset_1d = savitzky_golay(output_testset_1d, 25, 3) * 0.75

    return encoder_input_test_ppg_1d, encoder_input_test_ecg_1d, decoder_output_test_1d, smooth_output_testset_1d