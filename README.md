# Read Me

What is all this?
1) This project is an application of https://github.com/FlorentF9/DESOM for clustering MaNGA spectra, and subsequently clustering galaxies.
2) Computation was primarily done on ComputeCanada Cedar Cluster at SFU in 2020/21.
3) https://arxiv.org/abs/2112.03425


![MaNGA DESOM Pipeline](https://user-images.githubusercontent.com/76233047/134795641-758f833d-45bf-4fa8-8244-08586f2ba473.png)


How the code was run:

Step 1: Split data
There are ~1 million spectra from ~4_600 galaxies.
We identify and remove unsuitable spectra.
We use ~300_000 spectra (randomly sampled) for training.
There are 3 flavours of data in this project: raw spectra (for example, x_test.fits), wavelength (w_test.fits), and parameter data (y_test.fits).

Step 2: Run a model
Run a DESOM model that clusters these spectra with "DESOM_train.py". It runs a Conv1D model AutoEncoder with a SOM attached to the latent layer.
It also saves some diagnostic images.

Step 3: Update parameter data
Calculate sSFR, update logMS_dens using LambdaCDM correction, calculate where each spectrum lands on the bmu.

Step 3: Inspect DESOM-1 (notebook)
Look at prototypes, make sure whole map is used. Look at distribution of physical parameters.

Step 4: Generate Fingerprints
Loop through each galaxy, and collect the density of activated nodes.

Step 5: Train a DESOM-2 model
This is similar to the above model.

Step 6: Inspect DESOM-2 (notebook)
Look at DESOM-2 map, including morphology distribution, and Sersic.

Step 7: Get Galaxy Images (notebook)
Pull visible light images of galaxies using SDSS Marvin API.
