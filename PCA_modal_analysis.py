import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy import signal
import argparse
import os


def _transmissibility(signal_1, signal_2, sampling_frequency, noverlap=0, nperseg=256):
    freq, pyx1 = signal.csd(signal_2, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    freq, pxx1 = signal.csd(signal_1, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    transmis = np.abs(np.divide(pyx1, pxx1))
    return freq, transmis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testing_data", type=str, help="The path to the testing data", required=True)
    parser.add_argument("-l", "--linear_data", type=str, help="The path to save the linear-modal-analysis (PCA) data",
                        required=True)
    parser.add_argument("-f", "--sampling_frequency", type=float, help="The sampling frequency", default=100.0)
    args = parser.parse_args()

    ###########################################################################################################
    ##############                  Define the file paths                        ##############################
    ###########################################################################################################
    TRAINING_DATA_PATH = args.testing_data
    LINEAR_MODAL_ANALYSIS_DATA_PATH = args.linear_data
    # Set the parameters for plotting labels
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    NPERSEG = 4096 * 6
    sampling_freq = args.sampling_frequency

    # Load the file used for training and model selection
    file_path = os.path.join(os.getcwd(), TRAINING_DATA_PATH)
    disps = np.genfromtxt(file_path, delimiter=',')
    n_dofs = disps.shape[1]

    pca = PCA(n_components=n_dofs)
    pcaed_disps = pca.fit_transform(disps)

    PSDs = []
    for i in range(n_dofs):
        freq, psd = signal.welch(pcaed_disps[:, i], sampling_freq, noverlap=0, nperseg=NPERSEG)
        PSDs.append(psd)

    fig, axs = plt.subplots(1, n_dofs)
    for i in range(n_dofs):
        axs[i].plot(freq, PSDs[i])
    plt.show()

    np.savetxt(LINEAR_MODAL_ANALYSIS_DATA_PATH, pcaed_disps, delimiter=',')


if __name__ == "__main__":
    main()
