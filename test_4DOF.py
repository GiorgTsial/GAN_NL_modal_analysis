import numpy as np
from keras.models import load_model
import os
from matplotlib import pyplot as plt
from scipy import signal
import itertools
from sklearn.decomposition import PCA
import dcor
import seaborn as sn
import pandas as pd
from scipy.stats import pearsonr
import argparse


def _transmissibility(signal_1, signal_2, sampling_frequency, noverlap=0, nperseg=256):
    freq, pyx1 = signal.csd(signal_2, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    freq, pxx1 = signal.csd(signal_1, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    transmis = np.abs(np.divide(pyx1, pxx1))
    return freq, transmis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--training_data", type=str, help="The path to the training data", required=True)
    parser.add_argument("-t", "--testing_data", type=str, help="The path to the testing data", required=True)
    parser.add_argument("-l", "--linear_data", type=str, help="The path to the linear-modal-analysis data", required=True)
    parser.add_argument("-c", "--checkpoint_folder", type=str, help="The path to the model checkpoints", required=True)
    parser.add_argument("-f", "--sampling_frequency", type=float, help="The sampling frequency", default=100.0)
    args = parser.parse_args()
    ###########################################################################################################
    ##############                  Define the file paths                        ##############################
    ###########################################################################################################
    TRAINING_DATA_PATH = args.training_data
    TESTING_DATA_PATH = args.testing_data
    LINEAR_MODAL_ANALYSIS_DATA_PATH = args.linear_data
    CHECKPOINTS_PATH = args.checkpoint_folder

    ###########################################################################################################
    ##############                  Load data and plot physical PSDs             ##############################
    ###########################################################################################################

    # Set the parameters for plotting labels
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)

    NPERSEG = 4096 * 6
    sampling_freq = args.sampling_frequency

    # Load the file used for training and model selection
    file_path = TRAINING_DATA_PATH
    disps = np.genfromtxt(file_path, delimiter=',')

    # Plot the PSDs of the physical coordinates
    freqs, PSD1 = signal.welch(disps[:, 0], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, PSD2 = signal.welch(disps[:, 1], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, PSD3 = signal.welch(disps[:, 2], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, PSD4 = signal.welch(disps[:, 3], sampling_freq, noverlap=0, nperseg=NPERSEG)


    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(freqs, PSD1)
    axs[0, 1].plot(freqs, PSD2)
    axs[1, 0].plot(freqs, PSD3)
    axs[1, 1].plot(freqs, PSD4)

    axs[0, 0].set_xlim([-.01, 2.0])
    # axs[0, 0].set_title("PSD 1")
    axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 1].set_xlim([-.01, 2.0])
    # axs[0, 1].set_title("PSD 2")
    axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 0].set_xlim([-.01, 2.0])
    # axs[1, 0].set_title("PSD 3")
    axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 1].set_xlim([-.01, 2.0])
    # axs[1, 1].set_title("PSD 4")
    axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 0].set_ylabel("$Y_{1}(f)$", fontsize=16)
    axs[0, 1].set_ylabel("$Y_{2}(f)$", fontsize=16)
    axs[1, 0].set_ylabel("$Y_{3}(f)$", fontsize=16)
    axs[1, 1].set_ylabel("$Y_{4}(f)$", fontsize=16)

    plt.show()

    ###########################################################################################################
    ##############                  PCA the data and select best model             ############################
    ###########################################################################################################

    # Define the pca object and decompose/scale the displacements
    pca_before = PCA(n_components=4)
    pcaed_disps = pca_before.fit_transform(disps)
    pcaed_disps = 2 * (pcaed_disps - np.min(pcaed_disps, axis=0)) / (
                        np.max(pcaed_disps, axis=0) - np.min(pcaed_disps, axis=0)) - 1.0

    min_inner = 1000000
    best_encoder = None
    best_epoch = None

    # Look for the best cycleGAN
    # Best epoch is 300
    for epoch in range(0, 10000 + 1, 100):
        print("Testing epoch" + str(epoch))
        print("\n")
        encoder_path = os.path.join(os.getcwd(), CHECKPOINTS_PATH, "g_model_AtoB_" + str(epoch) + ".h5")
        if os.path.isfile(encoder_path):
            encoder = load_model(encoder_path)
            encoded_disps = encoder.predict(pcaed_disps)
            freqs, frf1 = signal.welch(encoded_disps[:, 0], sampling_freq, noverlap=0, nperseg=NPERSEG)
            freqs, frf2 = signal.welch(encoded_disps[:, 1], sampling_freq, noverlap=0, nperseg=NPERSEG)
            freqs, frf3 = signal.welch(encoded_disps[:, 2], sampling_freq, noverlap=0, nperseg=NPERSEG)
            freqs, frf4 = signal.welch(encoded_disps[:, 3], sampling_freq, noverlap=0, nperseg=NPERSEG)
            frfs = [frf1, frf2, frf3, frf4]
            combinations = list(itertools.combinations(range(4), 2))
            inner_prod = 0
            for comb in combinations:
                inner_prod += np.dot(frfs[comb[0]], frfs[comb[1]]) / (np.linalg.norm(frfs[comb[0]]) * np.linalg.norm(frfs[comb[1]]))
            if inner_prod < min_inner:
                min_inner = inner_prod
                best_encoder = encoder
                best_epoch = epoch


    print(best_epoch)
    print("Best inner prod: ", min_inner)

    ###########################################################################################################
    ##############          Load testing data and apply cycle-GAN decomposition             ###################
    ###########################################################################################################

    # Load the displacements used for testing, PCA them and scale them
    file_path = TESTING_DATA_PATH
    testing_disps = np.genfromtxt(file_path, delimiter=',')
    pcaed_disps = pca_before.transform(testing_disps)

    max_val = np.max(pcaed_disps, axis=0)
    min_val = np.min(pcaed_disps, axis=0)

    pcaed_disps = 2 * (pcaed_disps - np.min(pcaed_disps, axis=0)) / (
                        np.max(pcaed_disps, axis=0) - np.min(pcaed_disps, axis=0)) - 1.0

    # Cycle GAN decompose the displacements
    encoded_disps = best_encoder.predict(pcaed_disps)

    freqs, cG_frf1 = signal.welch(encoded_disps[:, 0], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, cG_frf2 = signal.welch(encoded_disps[:, 1], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs_cG, cG_frf3 = signal.welch(encoded_disps[:, 2], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, cG_frf4 = signal.welch(encoded_disps[:, 3], sampling_freq, noverlap=0, nperseg=NPERSEG)


    # Plot the comparision between the linear modal decomposition and the cycle-GAN

    # Plot 2x2 plot of cycle-GAN decomposition PSDs
    fig, axs = plt.subplots(2, 2)
    # fig.suptitle('Modal coordinates')
    axs[0, 0].plot(freqs_cG, cG_frf1, c="r")
    axs[1, 0].plot(freqs_cG, cG_frf2, c="r")
    axs[1, 1].plot(freqs_cG, cG_frf3, c="r")
    axs[0, 1].plot(freqs_cG, cG_frf4, c="r")

    axs[0, 0].set_xlim([-.01, 2.0])
    # axs[0, 0].set_title("PSD 1")
    axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 1].set_xlim([-.01, 2.0])
    # axs[0, 1].set_title("PSD 2")
    axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 0].set_xlim([-.01, 2.0])
    # axs[1, 0].set_title("PSD 3")
    axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 1].set_xlim([-.01, 2.0])
    # axs[1, 1].set_title("PSD 4")
    axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 0].set_ylabel("$U_{CG1}(f)$", fontsize=16)
    axs[0, 1].set_ylabel("$U_{CG2}(f)$", fontsize=16)
    axs[1, 0].set_ylabel("$U_{CG3}(f)$", fontsize=16)
    axs[1, 1].set_ylabel("$U_{CG4}(f)$", fontsize=16)

    plt.show()

    file_path = LINEAR_MODAL_ANALYSIS_DATA_PATH

    disps = np.genfromtxt(file_path, delimiter=',')

    freqs, frf1_linear_modal = signal.welch(disps[:, 0], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, frf2_linear_modal = signal.welch(disps[:, 1], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, frf3_linear_modal = signal.welch(disps[:, 2], sampling_freq, noverlap=0, nperseg=NPERSEG)
    freqs, frf4_linear_modal = signal.welch(disps[:, 3], sampling_freq, noverlap=0, nperseg=NPERSEG)


    fig, axs = plt.subplots(2, 2)
    # fig.suptitle('PCA coordinates')
    axs[0, 0].plot(freqs, frf1_linear_modal, c="k")
    axs[0, 1].plot(freqs, frf2_linear_modal, c="k")
    axs[1, 0].plot(freqs, frf3_linear_modal, c="k")
    axs[1, 1].plot(freqs, frf4_linear_modal, c="k")

    axs[0, 0].set_xlim([-.01, 2.0])
    # axs[0, 0].set_title("PSD 1")
    axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 1].set_xlim([-.01, 2.0])
    # axs[0, 1].set_title("PSD 2")
    axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 0].set_xlim([-.01, 2.0])
    # axs[1, 0].set_title("PSD 3")
    axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[1, 1].set_xlim([-.01, 2.0])
    # axs[1, 1].set_title("PSD 4")
    axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)

    axs[0, 0].set_ylabel("$U_{modal1}(f)$", fontsize=16)
    axs[0, 1].set_ylabel("$U_{modal2}(f)$", fontsize=16)
    axs[1, 0].set_ylabel("$U_{modal3}(f)$", fontsize=16)
    axs[1, 1].set_ylabel("$U_{modal4}(f)$", fontsize=16)

    plt.show()


    ###########################################################################################################
    ##############          Check the correlation of the modal coordinates                  ###################
    ###########################################################################################################

    # Calculate the correlations of the modal coordinates
    n_dofs = 4

    distance_corrs = np.zeros((n_dofs, n_dofs))
    encoded_disps = np.array(encoded_disps, dtype=np.float64)

    new_pca = PCA(n_components=n_dofs)
    encoded_disps_2 = new_pca.fit_transform(encoded_disps)

    # First the distance correlation
    for i in range(n_dofs):
        for j in range(n_dofs):
            distance_corrs[i, j] = dcor.distance_correlation(encoded_disps_2[:, i], encoded_disps_2[:, j], method="AVL")
    labels = ["1st Mode", "2nd Mode", "3rd Mode", "4th Mode"]
    df_cm_2 = pd.DataFrame(np.abs(distance_corrs), index=labels,
                         columns=labels)

    # Plot the distance correlation heat-map
    plt.figure()
    sn.heatmap(df_cm_2, annot=True, cmap="OrRd")
    plt.title("Distance correlation")
    plt.show()

    # Second the Pearson's correlation coefficient
    pearson_corr = np.zeros((n_dofs, n_dofs))
    for i in range(n_dofs):
        for j in range(n_dofs):
            pearson_corr[i, j] = pearsonr(encoded_disps_2[:, i], encoded_disps_2[:, j])[0]

    df_cm_1 = pd.DataFrame(np.abs(pearson_corr), index=labels,
                         columns=labels)

    # Plot the Pearson's correlation heat-map
    plt.figure()
    sn.heatmap(df_cm_1, annot=True, cmap="OrRd")
    plt.title("Pearson's correlation")
    plt.show()


    ###########################################################################################################
    ##############                      Perform the superposition step                      ###################
    ###########################################################################################################

    # Load the decoder and decode the encoded displacements
    decoder_path = os.path.join(os.getcwd(), CHECKPOINTS_PATH, "g_model_BtoA_" + str(best_epoch) + ".h5")
    decoder = load_model(decoder_path)
    decoded_disps = decoder.predict(encoded_disps)

    decoded_disps = (decoded_disps + 1) / 2 * (max_val - min_val) + min_val
    decoded_disps = pca_before.inverse_transform(decoded_disps)

    plt.plot(testing_disps[:, 0], c="blue")
    plt.plot(decoded_disps[:, 0], c="r")
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Displacement", fontsize=20)
    plt.show()

    errors = testing_disps - decoded_disps
    NMSE = 100 * np.sum(np.square(errors)) / (errors.shape[0] * errors.shape[1]) / np.std(testing_disps) ** 2
    print("NMSE of superposition = ", NMSE, "%")


if __name__ == "__main__":
    main()
