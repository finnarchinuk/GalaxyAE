##### Select a subset of spectra for training and testing.

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


NUM_SPECTRA = 400_000
SPEC_SIZE = 4544                 # number of data points for each spectrum
TRAIN_SPEC_PATH = 'X_train400k'
TEST_SPEC_PATH =  'X_test400k'

TRAIN_DATA_PATH = 'Y_train400k'
TEST_DATA_PATH =  'Y_test400k'

path='/my_favourite_data_path/MLdata/'

def get_samples(num_samples,seed=0,folder=path):
  np.random.seed(seed)
  #
  samples_per_set=int(np.floor(num_samples/10.))
  spec=np.zeros((samples_per_set*10,SPEC_SIZE))
  
  for i in range(10):                      # 10 files contain the data.
    temp_spec=np.load(folder+'spec'+str(i)+'.npy')
    my_info=Table.read(folder+'datatab'+str(i)+'.fits')
    #
    npick=samples_per_set
    ind_random = np.random.choice(np.arange(temp_spec.shape[0]),npick,replace=False)
    #
    dumb_mask=np.zeros(temp_spec.shape[0],dtype='bool')
    for k in range(npick):
        dumb_mask[ind_random[k]]=True
    #
    spec[samples_per_set*i:samples_per_set*i+samples_per_set]=temp_spec[dumb_mask]
    if i==0: info=my_info[dumb_mask]
    else: info=vstack([info,my_info[dumb_mask]])
  return spec,info
  
def train_test_split(spectra,data):
    #saves normalized spectra into training and test sets
    import os
    from sklearn import preprocessing
    from sklearn import model_selection as md
    np.random.seed(0)
    X = spectra
    Y = data

    #Standardize each ROW (normalization is within spectra, scaler doesn't give information.)
    scaler = preprocessing.StandardScaler().fit(X.T)
    scaled_data= scaler.transform(X.T)

    # To keep the parameters with the correct spectra, we're splitting a single column of the parameters ('SplitID')
    # then using "Y.loc" to recover the corresponding row. (Astropy has issues splitting a Table).
    Y['SplitID']=np.arange(len(Y))
    Y.add_index('SplitID')
    Xt,Xv,Yt,Yv = md.train_test_split(scaled_data.T,Y['SplitID'],test_size=0.1)
    #
    y_train=Y.loc[Yt]
    y_test= Y.loc[Yv]
    
    print('X_train,X_test shapes:',Xt.shape,Xv.shape)
    np.save(TRAIN_SPEC_PATH,Xt)
    y_train.write(TRAIN_DATA_PATH+'.fits',format='fits')
    np.save(TEST_SPEC_PATH,Xv)
    y_test.write(TEST_DATA_PATH+'.fits',format='fits')
    print ('Done!')    

# --------
# SELECT SPECTRA, SAVE IT
# --------
a,b=get_samples(NUM_SPECTRA) # 'a' are spectra, 'b' are the corresponding parameters.
train_test_split(spectra=a,data=b)


# ----------------
# Generate gal_unique_indexes.npy
# This is an array of the MaNGA suffixes which will later be used for looping through galaxies separately.
# ---------------

# ------- Load Full Datatab --------
full_datatab=Table.read(path+'datatab0.fits')
for i in range(1,10):
    full_datatab=vstack([full_datatab,Table.read(path+'datatab'+str(i)+'.fits')])
unique_galaxies=np.unique(full_datatab['mangaID'])

unique_suffixes=np.zeros(len(unique_galaxies)) #there are 4609 galaxies.
for i in range(len(unique_suffixes)):
    unique_suffixes[i]=unique_galaxies[i].split('-')[1] #removes the prefix and dash
unique_suffixes.sort()

np.save('gal_unique_indexes.npy',unique_suffixes)
print('saved gal_unique_indexes.npy')
