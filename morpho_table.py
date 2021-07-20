'''
VALUE ADDED CATALOGUE FROM: Vazquez-Mata and Hernandez-Toledo
https://www.sdss.org/dr16/data_access/value-added-catalogs/?vac_id=manga-visual-morphologies-from-sdss-and-desi-images

This is a small script that pulls galaxy information from a value added catalogue.
This table ends up having 4696 galaxies, though some are blanks.
'''

from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

# Load data
file_path = 'SOME_PATH/manga_visual_morpho-1.0.1.fits'
catalogue = fits.open(file_path)
vac_data = catalogue[1].data


# Create simplified table
column_names = ['mangaID', 'Type', 'edge_on', 'tidal', 'Conc', 'Conc_err', 'Asym', 'Asym_err', 'Clump', 'Clump_err']
vac_table = Table(names = column_names,
                  data = [vac_data.field('MANGAID'),
                          vac_data.field('Type'),
                          vac_data.field('edge_on'),
                          vac_data.field('tidal'),
                          vac_data.field('C'),
                          vac_data.field('E_C'),
                          vac_data.field('A'),
                          vac_data.field('E_A'),
                          vac_data.field('S'),
                          vac_data.field('E_S')])

vac_table.write('full vac.fits')
