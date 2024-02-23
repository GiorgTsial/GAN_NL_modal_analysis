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


def _transmissibility(signal_1, signal_2, sampling_frequency, noverlap=0, nperseg=256):
    freq, pyx1 = signal.csd(signal_2, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    freq, pxx1 = signal.csd(signal_1, signal_1, fs=sampling_frequency, noverlap=noverlap, nperseg=nperseg)
    transmis = np.abs(np.divide(pyx1, pxx1))
    return freq, transmis

###########################################################################################################
##############                  Define the file paths                        ##############################
###########################################################################################################
TRAINING_DATA_PATH = "data/3_DOF_nonlinear_disps.csv"
TESTING_DATA_PATH = "data/3_DOF_nonlinear_disps_test_dataset.csv"
LINEAR_MODAL_ANALYSIS_DATA_PATH = "data/3_DOF_nonlinear_disps_linear_modal.csv"
CHECKPOINTS_PATH = "checkpoints/cycleGAN_checkpoints_3DOF_nonlinear"

###########################################################################################################
##############                  Load data and plot physical PSDs             ##############################
###########################################################################################################

# Set the parameters for plotting labels
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

NPERSEG = 4096 * 6

# Load the file used for training and model selection
file_path = os.path.join(os.getcwd(), TRAINING_DATA_PATH)
disps = np.genfromtxt(file_path, delimiter=',')

# Plot the PSDs of the physical coordinates
freqs, PSD1 = signal.welch(disps[:, 0], 100, noverlap=0, nperseg=NPERSEG)
freqs, PSD2 = signal.welch(disps[:, 1], 100, noverlap=0, nperseg=NPERSEG)
freqs, PSD3 = signal.welch(disps[:, 2], 100, noverlap=0, nperseg=NPERSEG)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(freqs, PSD1)
ax2.plot(freqs, PSD2)
ax3.plot(freqs, PSD3)

ax1.set_xlim([-.01, 2.0])
# ax1.set_title("PSD 1")
ax1.set_xlabel("Frequency (Hz)", fontsize=16)

ax2.set_xlim([-.01, 2.0])
# ax2.set_title("PSD 2")
ax2.set_xlabel("Frequency (Hz)", fontsize=16)

ax3.set_xlim([-.01, 2.0])
# ax3.set_title("PSD 3")
ax3.set_xlabel("Frequency (Hz)", fontsize=16)

ax1.set_ylabel("$Y_{1}(f)$", fontsize=16)
ax2.set_ylabel("$Y_{2}(f)$", fontsize=16)
ax3.set_ylabel("$Y_{3}(f)$", fontsize=16)

plt.show()

###########################################################################################################
##############                  PCA the data and select best model             ############################
###########################################################################################################

# Define the pca object and decompose/scale the displacements
pca_before = PCA(n_components=3)
pcaed_disps = pca_before.fit_transform(disps)
pcaed_disps = 2 * (pcaed_disps - np.min(pcaed_disps, axis=0)) / (
                    np.max(pcaed_disps, axis=0) - np.min(pcaed_disps, axis=0)) - 1.0

min_inner = 1000000
best_encoder = None
best_epoch = None

# Look for the best cycleGAN
# Best epoch is 7000
for epoch in range(7000, 7000 + 1, 100):
    print("Testing epoch" + str(epoch))
    print("\n")
    encoder_path = os.path.join(os.getcwd(), CHECKPOINTS_PATH, "g_model_AtoB_" + str(epoch) + ".h5")
    encoder = load_model(encoder_path)
    encoded_disps = encoder.predict(pcaed_disps)
    freqs, frf1 = signal.welch(encoded_disps[:, 0], 100, noverlap=0, nperseg=NPERSEG/2)
    freqs, frf2 = signal.welch(encoded_disps[:, 1], 100, noverlap=0, nperseg=NPERSEG/2)
    freqs, frf3 = signal.welch(encoded_disps[:, 2], 100, noverlap=0, nperseg=NPERSEG/2)
    frfs = [frf1, frf2, frf3]
    combinations = list(itertools.combinations(range(3), 2))
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
file_path = os.path.join(os.getcwd(), TESTING_DATA_PATH)
testing_disps = np.genfromtxt(file_path, delimiter=',')
pcaed_disps = pca_before.transform(testing_disps)

max_val = np.max(pcaed_disps, axis=0)
min_val = np.min(pcaed_disps, axis=0)

pcaed_disps = 2 * (pcaed_disps - np.min(pcaed_disps, axis=0)) / (
                    np.max(pcaed_disps, axis=0) - np.min(pcaed_disps, axis=0)) - 1.0

# Cycle GAN decompose the displacements
encoded_disps = best_encoder.predict(pcaed_disps)

freqs, cG_frf1 = signal.welch(encoded_disps[:, 0], 100, noverlap=0, nperseg=NPERSEG)
freqs, cG_frf2 = signal.welch(encoded_disps[:, 1], 100, noverlap=0, nperseg=NPERSEG)
freqs_cG, cG_frf3 = signal.welch(encoded_disps[:, 2], 100, noverlap=0, nperseg=NPERSEG)


# Plot the comparision between the linear modal decomposition and the cycle-GAN
fig, axs = plt.subplots(2, 3)
# fig.suptitle('"Modal" coordinates')
axs[1, 0].plot(freqs, cG_frf1, c="r")
axs[1, 1].plot(freqs, cG_frf2, c="r")
axs[1, 2].plot(freqs, cG_frf3, c="r")

axs[1, 0].set_xlim([-.01, 2.0])
# axs[1, 0].set_title("PSD 1")
axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)

axs[1, 1].set_xlim([-.01, 2.0])
# axs[1, 1].set_title("PSD 2")
axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)

axs[1, 2].set_xlim([-.01, 2.0])
# axs[1, 2].set_title("PSD 3")
axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=16)

axs[1, 0].set_ylabel("$U_{CG1}(f)$", fontsize=16)
axs[1, 1].set_ylabel("$U_{CG2}(f)$", fontsize=16)
axs[1, 2].set_ylabel("$U_{CG3}(f)$", fontsize=16)

file_path = os.path.join(os.getcwd(), LINEAR_MODAL_ANALYSIS_DATA_PATH)

disps = np.genfromtxt(file_path, delimiter=',')

freqs, frf1_linear_modal = signal.welch(disps[:, 0], 100, noverlap=0, nperseg=NPERSEG)
freqs, frf2_linear_modal = signal.welch(disps[:, 1], 100, noverlap=0, nperseg=NPERSEG)
freqs, frf3_linear_modal = signal.welch(disps[:, 2], 100, noverlap=0, nperseg=NPERSEG)

# fig.suptitle('Modal coordinates')
axs[0, 0].plot(freqs, frf1_linear_modal, c="k")
axs[0, 1].plot(freqs, frf2_linear_modal, c="k")
axs[0, 2].plot(freqs, frf3_linear_modal, c="k")

axs[0, 0].set_xlim([-.01, 2.0])
# axs[0, 0].set_title("PSD 1")
axs[0, 0].set_xlabel("Frequency (Hz)")

axs[0, 1].set_xlim([-.01, 2.0])
# axs[0, 1].set_title("PSD 2")
axs[0, 1].set_xlabel("Frequency (Hz)")

axs[0, 2].set_xlim([-.01, 2.0])
# axs[0, 2].set_title("PSD 3")
axs[0, 2].set_xlabel("Frequency (Hz)")

axs[0, 0].set_ylabel("$Y_{modal1}(f)$", fontsize=16)
axs[0, 1].set_ylabel("$Y_{modal2}(f)$", fontsize=16)
axs[0, 2].set_ylabel("$Y_{modal3}(f)$", fontsize=16)

plt.show()


###########################################################################################################
##############          Check the correlation of the modal coordinates                  ###################
###########################################################################################################

# Calculate the correlations of the modal coordinates
n_dofs = 3

distance_corrs = np.zeros((n_dofs, n_dofs))
encoded_disps = np.array(encoded_disps, dtype=np.float64)

# First the distance correlation
for i in range(n_dofs):
    for j in range(n_dofs):
        distance_corrs[i, j] = dcor.distance_correlation(encoded_disps[:, i], encoded_disps[:, j], method="AVL")
labels = ["1st Mode", "2nd Mode", "3rd Mode"]
df_cm_2 = pd.DataFrame(np.abs(distance_corrs), index=labels,
                     columns=labels)

# Plot the distance correlation heat-map
plt.figure()
sn.heatmap(df_cm_2, annot=True, cmap="OrRd")
plt.title("Distance correlation")
plt.show()

# First the Pearson's correlation coefficient
pearson_corr = np.zeros((n_dofs, n_dofs))
for i in range(n_dofs):
    for j in range(n_dofs):
        pearson_corr[i, j] = pearsonr(encoded_disps[:, i], encoded_disps[:, j])[0]

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