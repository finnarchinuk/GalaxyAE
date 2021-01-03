# Read Me

What is all this?
1) This project is an application of https://github.com/FlorentF9/DESOM for clustering MaNGA spectra, and subsequently using these clusters for clustering similar galaxies.
2) Computation was primarily done on ComputeCanada Cedar Cluster at SFU in 2020.
3) The corresponding paper is being written (as of Jan 2021)

How the code was run:

Step 1: Split data (select_spectra_jul18.py)
There are ~1 million spectra from ~4_600 galaxies. We use ~200_000 spectra randomly sampled from these 4_600 galaxies for training.
Data used in this project are in two types: the spectra and parameters. Parameters are pulled from spectra or are metadata from observation (such as r_re_light).
The script to load this is "select_spectra_jul18.py". It pulls out some number of spectra from each of the 10 files.
Spectra are Standard Scaled here (along their length).
This saves some files: spec250k.npy, spec250k_info.fits. These two files are further split into X_train.npy, X_test.npy, Y_train.fits, Y_test.fits.

Step 2: Run a model (DESOM_train.py)
Run a model that clusters these spectra with "DESOM_train.py". It runs a Conv1D model Autoencoder with a SOM output. The weights are saved in 'results/tmp/$arg/DESOM_model_final.h5'. This also saves some quick and dirty diagnostics (anything that ends in .png)

Step 3: Interrogate DESOM-1 (STEP3_DESOM1.ipynb)
Look at prototypes, make sure whole map is used. Look at other stuff. Generally check that the DESOM1 model works.

Step 4a: Split spectra by galaxy (Generate_Fingerprints.ipynb)
This goes into the original data, split by galaxy, save these to 'data/gal_split_100/'. Saves the galaxies as '<manga_suffix>.npy', and the data as '<manga_suffix>_info.fits'.
This only needs to be run once.
(Note: MaNGA IDs are something like '1-44219'. 'suffix' refers to '44219'. These suffixes are sufficient to distinguish galaxies.)
  
Step 4b: Use the model to generate fingerprints
This loads a model. Then it loads the spectra of a galaxy and plots it on the SOM grid. These SOM activations are flattened into a one dimensional 'histo-vector' with length equal to the number of nodes in the SOM. These fingerprints are saved as a '.npy' file that can be used for training DESOM2. (We save these fingerprints as 1-dimesional data because that's where we started, but DESOM-2 reshapes this data back into a 2D-fingerprint).

Step 5a: train DESOM-2 using fingerprints (currently overleaf_part2.ipynb)
(Trained on a gpu for speed instead of in this)
This is effectively the same as DESOM_train.py, but the code was copied for adjusting and finding better training parameters. Trained on ~4_000 fingerprints (one per galaxy), tested on the ~600 remainder.

Step 5b: Interrogate DESOM-2
Plots things, looks at systematic error.



#### generate fingerprints
# fix this to run using concurrent.
