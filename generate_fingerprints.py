'''
runs each galaxy through a DESOM model, determines where each spectrum lands
also adds on a morphology for that galaxy.
'''


print('\n\n generating fingerprints')

from astropy.table import Table, vstack
import numpy as np


# load data and tables
SUFFIX = 'june11bmu'
MAP_SIZE = 15 * 15

gal_morph = Table.read('gal_morph.fits') # generated from a value added catalogue (see morpho_table.py)
my_train = Table.read('y_train1m_'+SUFFIX+'.fits')
my_test = Table.read('y_test1m_'+SUFFIX+'.fits')
my_unused = Table.read('y_unused1m_'+SUFFIX+'.fits')


# get a list of all mangaIDs
def pull_unique_gal_indexes(data):
  ''' makes a list of all galaxies
      saveable as file, or
      return unique_ids, and unique_suffixes
  '''
  unique_ids = np.unique(data['mangaID'])
  unique_suffixes = np.zeros(len(unique_ids))
  for i in range(len(unique_ids)):
    unique_suffixes[i]=unique_ids[i].split('-')[1]
  return unique_ids, unique_suffixes

unique_ids, unique_suffixes = pull_unique_gal_indexes(vstack([my_train, my_test, my_unused]))


train_fp = np.zeros((len(unique_ids), MAP_SIZE))
test_fp = np.zeros((len(unique_ids), MAP_SIZE))
unused_fp = np.zeros((len(unique_ids), MAP_SIZE))

# generate FPs for each galaxy
# (it's not important to keep train/test/unused spectra separate, just more because I can)
for gal_idx, gal_id in enumerate(unique_ids):
  mask_train = (my_train['mangaID'] == gal_id)
  mask_test = (my_test['mangaID'] == gal_id)
  mask_unused = (my_unused['mangaID'] == gal_id)
  #
  temp_train = my_train[mask_train]['bmu']
  temp_test = my_test[mask_test]['bmu']
  temp_unused = my_unused[mask_unused]['bmu']
  for BMU in range(MAP_SIZE):
    hmask_train = (temp_train == BMU)
    train_fp[gal_idx, BMU] = hmask_train.sum()

    hmask_test = (temp_test == BMU)
    test_fp[gal_idx, BMU] = hmask_test.sum()

    hmask_unused = (temp_unused == BMU)
    unused_fp[gal_idx, BMU] = hmask_unused.sum()

# incorporate morphology list
morph_list = list()
for m in unique_ids:
  mask = (gal_morph['mangaID']==m)
  morph_list.append(gal_morph['type'][mask][0])

# define table
FP_table = Table([unique_ids,morph_list,train_fp,test_fp,unused_fp],
                 names=['mangaID','morph','train_fp','test_fp','unused_fp'])

# save fingerprints
FP_table.write('RF_fingerprint_faster.fits')

