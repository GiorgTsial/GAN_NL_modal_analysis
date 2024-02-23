import numpy as np
from keras.models import load_model
import os
from matplotlib import pyplot as plt
from scipy import io, signal
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


NPERSEG = 2048 * 2

mat = io.loadmat("state12.mat")
temp = mat["data"]
force = temp[:, 0, :]
disps = temp[:, 2:, :]

freqs, frf1 = signal.welch(disps[:, 0, 0], 320, noverlap=0, nperseg=NPERSEG)
freqs, frf2 = signal.welch(disps[:, 1, 0], 320, noverlap=0, nperseg=NPERSEG)
freqs, frf3 = signal.welch(disps[:, 2, 0], 320, noverlap=0, nperseg=NPERSEG)

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

print(len(disps[1000:, 0, 0]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Original coordinates')
ax1.plot(freqs, frf1)
ax2.plot(freqs, frf2)
ax3.plot(freqs, frf3)

# ax1.set_title("PSD 1")
ax1.set_xlabel("Frequency (Hz)", fontsize=16)
ax1.set_xlim([-.01, 100.0])

# ax2.set_title("PSD 2")
ax2.set_xlabel("Frequency (Hz)", fontsize=16)
ax2.set_xlim([-.01, 100.0])

# ax3.set_title("PSD 3")
ax3.set_xlabel("Frequency (Hz)", fontsize=16)
ax3.set_xlim([-.01, 100.0])

ax1.set_ylabel("$Y_{1}(f)$", fontsize=16)
ax2.set_ylabel("$Y_{2}(f)$", fontsize=16)
ax3.set_ylabel("$Y_{3}(f)$", fontsize=16)

plt.show()

pca_before = PCA(n_components=3)

disps_all = np.genfromtxt("los_alamos_state_12_all_accels.csv", delimiter=',')

pcaed_all_disps = pca_before.fit_transform(disps_all)

min_val = np.min(pcaed_all_disps, axis=0)
max_val = np.max(pcaed_all_disps, axis=0)

print(disps.shape)
pcaed_disps = []
for i in range(50):
    temp = pca_before.transform(disps[:, :, i]).T
    pcaed_disps.append(temp)

pcaed_disps = np.array(pcaed_disps).T

# disps = 2 * (pcaed_disps - np.min(pcaed_disps, axis=0)) / (
#                     np.max(pcaed_disps, axis=0) - np.min(pcaed_disps, axis=0)) - 1.0

for i in range(3):
    disps[:, i, :] = 2 * (pcaed_disps[:, i, :] - min_val[i]) / (max_val[i] - min_val[i]) - 1.0

force = 2 * (force - np.min(force, axis=0)) / (
                    np.max(force, axis=0) - np.min(force, axis=0)) - 1.0

print(disps.shape)
min_inner = 1000000000000000000000
best_encoder = None
best_epoch = 0

# Best epoch is 8100
for epoch in range(8100, 8100 + 1, 100):
    print("Testing epoch" + str(epoch))
    print("\n")
    encoder_path = os.path.join(os.getcwd(), "cycle_GAN_checkpoints_nonlinear_LANL_bookshelf_PCA", "g_model_AtoB_" + str(epoch) + ".h5")
    encoder = load_model(encoder_path)
    inner_prod = 0
    for i in range(45):
        encoded_disps = encoder.predict(disps[:, :, i])

        encoded_disps = encoded_disps - np.mean(encoded_disps, axis=0)

        freqs, frf1 = signal.welch(encoded_disps[:, 0], 320, noverlap=0, nperseg=NPERSEG)
        freqs, frf2 = signal.welch(encoded_disps[:, 1], 320, noverlap=0, nperseg=NPERSEG)
        freqs, frf3 = signal.welch(encoded_disps[:, 2], 320, noverlap=0, nperseg=NPERSEG)
        frfs = [frf1, frf2, frf3]
        combinations = list(itertools.combinations(range(3), 2))

        for comb in combinations:
            inner_prod += np.dot(frfs[comb[0]], frfs[comb[1]]) / (np.linalg.norm(frfs[comb[0]]) * np.linalg.norm(frfs[comb[1]]))
    inner_prod /= 50
    if inner_prod < min_inner:
        best_epoch = epoch
        min_inner = inner_prod
        best_encoder = encoder


print("Best inner product: ", min_inner)
print("Best epoch: ", best_epoch)

sum_frf1 = 0
sum_frf2 = 0
sum_frf3 = 0

all_encoded_disps = []

for i in range(50):
    encoded_disps = best_encoder.predict(disps[:, :, i])

    all_encoded_disps.append(encoded_disps)

    freqs, frf1 = signal.welch(encoded_disps[:, 0], 320, noverlap=0, nperseg=NPERSEG)
    freqs, frf2 = signal.welch(encoded_disps[:, 1], 320, noverlap=0, nperseg=NPERSEG)
    freqs, frf3 = signal.welch(encoded_disps[:, 2], 320, noverlap=0, nperseg=NPERSEG)

    sum_frf1 += frf1
    sum_frf2 += frf2
    sum_frf3 += frf3

mean_frf1 = sum_frf1 / 50
mean_frf2 = sum_frf2 / 50
mean_frf3 = sum_frf3 / 50

fig, axs = plt.subplots(2, 3)
fig.suptitle('"Modal" coordinates')
axs[0, 0].plot(freqs, mean_frf1, c="r")
axs[0, 1].plot(freqs, mean_frf2, c="r")
axs[0, 2].plot(freqs, mean_frf3, c="r")

# axs[0, 0].set_title("PSD 1")
axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 0].set_xlim([0.0, 100.0])

# axs[0, 1].set_title("PSD 2")
axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 1].set_xlim([0.0, 100.0])

# axs[0, 2].set_title("PSD 3")
axs[0, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 2].set_xlim([0.0, 100.0])

axs[0, 0].set_ylabel("Amplitude")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 2].set_ylabel("Amplitude")


all_accels = np.genfromtxt("los_alamos_state_12_all_accels.csv", delimiter=',')

pca = PCA(n_components=3)

pca.fit_transform(all_accels)

for i in range(50):
    encoded_disps = pca.transform(disps[:, :, i])

    # encoded_disps = encoded_disps - np.mean(encoded_disps, axis=0)

    freqs, frf1 = signal.welch(encoded_disps[:, 0], 320, noverlap=0, nperseg=NPERSEG)
    freqs, frf2 = signal.welch(encoded_disps[:, 1], 320, noverlap=0, nperseg=NPERSEG)
    freqs, frf3 = signal.welch(encoded_disps[:, 2], 320, noverlap=0, nperseg=NPERSEG)

    sum_frf1 += frf1
    sum_frf2 += frf2
    sum_frf3 += frf3

mean_frf1_PCA = sum_frf1 / 50
mean_frf2_PCA = sum_frf2 / 50
mean_frf3_PCA = sum_frf3 / 50


axs[1, 0].plot(freqs, mean_frf1_PCA, c="k")
axs[1, 1].plot(freqs, mean_frf2_PCA, c="k")
axs[1, 2].plot(freqs, mean_frf3_PCA, c="k")

# axs[1, 0].set_title("PSD 1")
axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 0].set_xlim([0.0, 100.0])

# axs[1, 1].set_title("PSD 2")
axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 1].set_xlim([0.0, 100.0])

# axs[1, 2].set_title("PSD 3")
axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 2].set_xlim([0.0, 100.0])

axs[1, 0].set_ylabel("Amplitude")
axs[1, 1].set_ylabel("Amplitude")
axs[1, 2].set_ylabel("Amplitude")

plt.show()

params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

fig, axs = plt.subplots(2, 3)
# fig.suptitle('"Modal" coordinates')
axs[1, 0].plot(freqs, mean_frf1, c="r")
axs[1, 1].plot(freqs, mean_frf2, c="r")
axs[1, 2].plot(freqs, mean_frf3, c="r")

# axs[1, 0].set_title("PSD 1")
axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 0].set_xlim([0.0, 100.0])

# axs[1, 1].set_title("PSD 2")
axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 1].set_xlim([0.0, 100.0])

# axs[1, 2].set_title("PSD 3")
axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 2].set_xlim([0.0, 100.0])

axs[1, 0].set_ylabel("$Y_{1}(f)$", fontsize=16)
axs[1, 1].set_ylabel("$Y_{2}(f)$", fontsize=16)
axs[1, 2].set_ylabel("$Y_{3}(f)$", fontsize=16)

axs[0, 0].plot(freqs, mean_frf1_PCA, c="k")
axs[0, 1].plot(freqs, mean_frf2_PCA, c="k")
axs[0, 2].plot(freqs, mean_frf3_PCA, c="k")

# axs[0, 0].set_title("PSD 1")
axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 0].set_xlim([0.0, 100.0])

# axs[0, 1].set_title("PSD 2")
axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 1].set_xlim([0.0, 100.0])

# axs[0, 2].set_title("PSD 3")
axs[0, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 2].set_xlim([0.0, 100.0])

axs[0, 0].set_ylabel("$U_{1}(f)$", fontsize=16)
axs[0, 1].set_ylabel("$U_{1}(f)$", fontsize=16)
axs[0, 2].set_ylabel("$U_{1}(f)$", fontsize=16)

plt.show()


fig, axs = plt.subplots(2, 3)
axs[1, 2].plot(freqs, mean_frf1, c="r")
axs[1, 0].plot(freqs, mean_frf2, c="r")
axs[1, 1].plot(freqs, mean_frf3, c="r")

# axs[1, 0].set_title("PSD 1")
axs[1, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 0].set_xlim([0.0, 100.0])

# axs[1, 1].set_title("PSD 2")
axs[1, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 1].set_xlim([0.0, 100.0])

# axs[1, 2].set_title("PSD 3")
axs[1, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[1, 2].set_xlim([0.0, 100.0])

axs[1, 0].set_ylabel("$U_{CG1}(f)$", fontsize=16)
axs[1, 1].set_ylabel("$U_{CG2}(f)$", fontsize=16)
axs[1, 2].set_ylabel("$U_{CG3}(f)$", fontsize=16)


axs[0, 0].plot(freqs, mean_frf1_PCA, c="k")
axs[0, 1].plot(freqs, mean_frf2_PCA, c="k")
axs[0, 2].plot(freqs, mean_frf3_PCA, c="k")

# axs[0, 0].set_title("PSD 1")
axs[0, 0].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 0].set_xlim([0.0, 100.0])

# axs[0, 1].set_title("PSD 2")
axs[0, 1].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 1].set_xlim([0.0, 100.0])

# axs[0, 2].set_title("PSD 3")
axs[0, 2].set_xlabel("Frequency (Hz)", fontsize=16)
axs[0, 2].set_xlim([0.0, 100.0])

axs[0, 0].set_ylabel("$U_{PCA1}(f)$", fontsize=16)
axs[0, 1].set_ylabel("$U_{PCA2}(f)$", fontsize=16)
axs[0, 2].set_ylabel("$U_{PCA3}(f)$", fontsize=16)

plt.show()

# Calculate correlations

n_dofs = 3

distance_corrs = np.zeros((n_dofs, n_dofs))
all_encoded_disps = np.vstack(all_encoded_disps)
all_encoded_disps = np.array(all_encoded_disps, dtype=np.float64)

print(all_encoded_disps.shape)

# First the distance correlation
for i in range(n_dofs):
    for j in range(n_dofs):
        distance_corrs[i, j] = dcor.distance_correlation(all_encoded_disps[:, i], all_encoded_disps[:, j], method="AVL")
labels = ["1st Mode", "2nd Mode", "3rd Mode"]

df_cm_2 = pd.DataFrame(np.abs(distance_corrs), index=labels,
                     columns=labels)

plt.figure()
sn.heatmap(df_cm_2, annot=True, cmap="OrRd")
plt.show()

# Second the Pearson's correlation coefficient
pearson_corr = np.zeros((n_dofs, n_dofs))
for i in range(n_dofs):
    for j in range(n_dofs):
        pearson_corr[i, j] = pearsonr(all_encoded_disps[:, i], all_encoded_disps[:, j])[0]

df_cm_1 = pd.DataFrame(np.abs(pearson_corr), index=labels,
                     columns=labels)

# Plot the Pearson's correlation heat-map
plt.figure()
sn.heatmap(df_cm_1, annot=True, cmap="OrRd")
plt.show()

###########################################################################################################
##############                      Perform the superposition step                      ###################
###########################################################################################################

# Load the decoder and decode the encoded displacements
testing_disps = np.genfromtxt("los_alamos_state_12_all_accels.csv", delimiter=',')
decoder_path = os.path.join(os.getcwd(), "cycle_GAN_checkpoints_nonlinear_LANL_bookshelf_PCA", "g_model_BtoA_" + str(best_epoch) + ".h5")
decoder = load_model(decoder_path)
decoded_disps = decoder.predict(all_encoded_disps)

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