##### Select a subset of spectra for training and testing.

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import os
from sklearn import preprocessing
from sklearn import model_selection as md

NUM_SPECTRA = 1_000_000
SPEC_SIZE = 4544                 # number of data points for each spectrum

# where the original uncleaned data exists as 10 files, each with 117100 spectra and
# corresponding wavelengths and data. (spec0.npy, data0.fits, wave0.npy)
original_data_path='/project/MANGA/dr15/MLdata/'

#-------------------------- Define where to save the data ------------------
new_data_path='/my_favourite_data_path/MLdata/'
TRAIN_SPEC_PATH = 'X_train_1m'
TEST_SPEC_PATH =  'X_test1_m'

TRAIN_DATA_PATH = 'Y_train_1m'
TEST_DATA_PATH =  'Y_test_1m'

#----------------------- Calculate and apply masking -------------------------
# Load all spectra, find top and bottom flux outliers (top and bottom 0.5%, this may be on the low end).
# If spectra have any values above the max or below the min, we don't include them in train or test data.
def flux_outlier(load_path,outlier=0.5)
  ''' Calculates the top and bottom {outlier} percentile of the flux to be masked off'''
  x_maxes = np.zeros(0)
  x_mins = np.zeros(0)
  for i in range(10):
    temp_spec = np.load('data/outlier_removed_perc2/spec{}.npy'.format(i))
    x_maxes = np.append(x_maxes,temp_spec.max(axis=1))
    x_mins = np.append(x_mins,temp_spec.min(axis=1))
  return np.percentile(x_maxes,100-outlier), np.percentile(x_mins,outlier)

# More traditional masking, based on values in the spectral data information (eg, data0.fits).
def apply_mask(load_folder,save_folder):
  for i in range(10):                      # 10 datasets
    # load each block
    temp_spec=np.load(load_folder+'spec'+str(i)+'.npy')
    my_info=Table.read(load_folder+'datatab'+str(i)+'.fits')
    
    # defined masks, these are explained further in the paper.
    ngp_mask = (my_info['ngoodpixels'] > 3700)
    signal_mask = (my_info['signal'] < 3)
    chi2_mask = (my_info['chi2'] < 50)
    meansn2_mask = (my_info['meansn2'] > 5)

    # extract maxes of each spectrum
    spec_maxes = temp_spec.max(axis=1)
    max_mask=(spec_maxes < max_threshold_mask)
    # extract mins
    spec_mins = temp_spec.min(axis=1)
    min_mask=(spec_mins > min_threshold_mask)

    # put all the masks together
    total_mask = ngp_mask * signal_mask * chi2_mask * meansn2_mask * max_mask * min_mask
    # print how much each mask removed (there will be overlap)
    print('ngp_mask:',temp_spec.shape[0]-ngp_mask.sum())
    print('signal_mask:',temp_spec.shape[0]-signal_mask.sum())
    print('chi2_mask:',temp_spec.shape[0]-chi2_mask.sum())
    print('meansn2_mask:',temp_spec.shape[0]-meansn2_mask.sum())
    print('max_mask:',temp_spec.shape[0]-max_mask.sum())
    print('min_mask:',temp_spec.shape[0]-min_mask.sum())

    print('chunk '+str(i)+' masked. '+str(total_mask.sum())+' remain. ('+str(total_mask.sum()/total_mask.shape[0])+').')

    # save these cleaned data.
    np.save(save_folder+'spec'+str(i)+'.npy',temp_spec[total_mask])
    my_info[total_mask].write(save_folder+'datatab'+str(i)+'.fits',format='fits')

#------------------------- Define how to select spectra randomly from original data ------------------
def get_samples(num_samples,load_folder,seed=0):
  ''' Take spectral samples from each data chunk.
  Return: spectra (as numpy array), and data (as astropy table).
  '''
  np.random.seed(seed)
  samples_per_set=int(np.floor(num_samples/10.))
  spec=np.zeros((samples_per_set*10,SPEC_SIZE))
  
  for i in range(10):                      # 10 files contain the data.
    temp_spec=np.load(load_folder+'spec'+str(i)+'.npy')
    my_info=Table.read(load_folder+'datatab'+str(i)+'.fits')
    
    # Randomly select spectra
    ind_random = np.random.choice(np.arange(temp_spec.shape[0]),samples_per_set,replace=False)

    # (taking that randomly selected slice from an astropy table is more complicated)
    dumb_mask=np.zeros(temp_spec.shape[0],dtype='bool')
    for k in range(samples_per_set):
        dumb_mask[ind_random[k]]=True

    # Stack samples into a single variable
    spec[samples_per_set*i:samples_per_set*i+samples_per_set]=temp_spec[dumb_mask]
    if i==0: info=my_info[dumb_mask]
    else: info=vstack([info,my_info[dumb_mask]])
  return spec,info

# --------------------- Define how to split data into training and testing sets ------------------
def train_test_split(spectra, data, test_ratio=0.1):
    ''' Splits data into training and testing sets '''
    np.random.seed(0)
    X = spectra
    Y = data

    #Standardize each ROW (normalization is within spectra, therefore the scaler doesn't cause information leakage.)
    scaler = preprocessing.StandardScaler().fit(X.T)
    scaled_data= scaler.transform(X.T)

    # To keep the parameters with the correct spectra, we're splitting a single column of the parameters ('SplitID')
    # then using "Y.loc" to recover the corresponding row. (Astropy has issues splitting a Table).
    Y['SplitID']=np.arange(len(Y))
    Y.add_index('SplitID')
    Xt,Xv,Yt,Yv = md.train_test_split(scaled_data.T,Y['SplitID'],test_size=test_ratio)
    #
    y_train=Y.loc[Yt]
    y_test= Y.loc[Yv]
    
    print('X_train,X_test shapes:',Xt.shape,Xv.shape)
    np.save(TRAIN_SPEC_PATH,Xt)
    y_train.write(TRAIN_DATA_PATH+'.fits',format='fits')
    np.save(TEST_SPEC_PATH,Xv)
    y_test.write(TEST_DATA_PATH+'.fits',format='fits')
    print ('Done!')    

#------------------- Call these functions -----------------
# calculate outlier threshold
max_threshold_mask, min_threshold_mask = flux_outliers(original_data_path, outlier=0.5)

# remove spectra from consideration, resave them by chunk
apply_mask(original_data_path, new_data_path)

# randomly select spectral samples (first output), and corresponding information (second output)
quick_spec, quick_data = get_samples(NUM_SPECTRA, new_data_path)

# split into train and test sets (while keeping the astropy table uses paths from top of this file)
train_test_split(quick_spec, quick_data, test_ratio=0.1)


#--------------------- Generate list of galaxy indexes ---------------------
# This is an array of the MaNGA suffixes which will later be used for looping through galaxies separately.

# Load Full Datatab
full_datatab=Table.read(path+'datatab0.fits')
for i in range(1,10):
    full_datatab=vstack([full_datatab,Table.read(path+'datatab'+str(i)+'.fits')])
unique_galaxies=np.unique(full_datatab['mangaID'])

# 
unique_suffixes=np.zeros_like(unique_galaxies) #there are 4609 galaxies.
for i, galaxy_index in enumerate(unique_galaxies):
    unique_suffixes[i]=galaxy_index.split('-')[1] #removes the prefix and dash
unique_suffixes.sort() #the order doesn't actually matter here.

np.save('gal_unique_indexes.npy',unique_suffixes)
print('saved gal_unique_indexes.npy')
