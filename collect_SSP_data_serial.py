#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:34:55 2023

@author: dori

This script collect the single scattering data from the individual scattering
calculations and puts them in a netCDF file for faster access.
It is developed for the file system structure used within OPTIMICe, and the
projects PROM-PRISTINE, PROM-IMPRINT and PROM-FRAGILE.
The current version is designed to collect all the data for 1 frequency
such that it can be run in parallell for many frequencies and the data can be
collected later into a unique dataset. It probably would need a refresh once
there will be no single particle per size.

It is designed to be run with arguments, but you can of course set variables
The data is searched in a filesystem that looks like this:
    /<basename>/<size>/<frequency>/<orientation>/data
where:
    <basename> is a path to a root folder that contains all the data for many particles
    <size> is a string that holds some size information, currently in this fixed format ??.???mm
    <frequency> is a string that holds frequency information currently in this fixed format ???.?GHz
    <orientation> is a free format folder
check if this is consistent with your file system

It is designed to work on a single frequency, but multiple outputs can be combined along the frequency dimension
in the end it will produce a .nc file named <output_name><frequency>
"""

import os
import socket
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from sys import argv

scriptname, freq, basename, parttype, output_name = argv

#dipole_resolution = 20.0e-6 # now read from one logfile - meters 20 microns 
ice_density = 917.0 # kg/m3
freq = float(freq) # GHz
col_ampl = ['theta', 's1r', 's1i', 's2r', 's2i', 's3r', 's3i', 's4r', 's4i']

###############################
# Define where to find the data
###############################
freqstr = '{:05.1f}GHz/'.format(freq)

#basename = '/work/bb1314/single_crystals/thin_2momden_with#/'
#basename = '/work/bb1314/plateden_results/lambda5e-5to0.001/1to100monomers/'
#basename='/work/bb1314/single_crystals/small_plate_den/'

den_fld = '{}??.????mm/'.format(basename)
shpflds=sorted(glob(den_fld))

##############################################################
# Define the dimensions and coordinates of the netcdf database
##############################################################
Dmax = np.array([float(s.split('mm')[0].split('/')[-1]) for s in shpflds])
sizes = xr.IndexVariable(dims='Dmax', data=Dmax,
                         attrs={'name':'Dmax',
                                'long_name':'Size - Maximum dimension',
                                'units':'millimeters'})

frequencies = xr.IndexVariable(dims='frequency', data=[freq],
                               attrs={'name':'freq',
                                      'long_name':'frequency of the electromagnetic wave',
                                      'units':'GigaHertz'})

Nangles = 2562  # TODO Max number of sampled directions
angles = xr.IndexVariable(dims='direction',
                          data=np.arange(0, Nangles),
                          attrs={'name':'a_idx',
                                 'long_name':'scattering direction index',
                                 'units':'dimensionless'})
                          
dims = ['Dmax', 'frequency', 'direction']
coords = {'Dmax':sizes, 'frequency':frequencies, 'direction':angles}

##############################################################
# Wavelength and ref index as variable for sanity check
##############################################################
wavelengths = xr.DataArray(dims=['frequency'], coords={'frequency':frequencies},
                           attrs={'name':'wl',
                                  'long_name':'wavelength',
                                  'units':'meters'})
ref_idx_r = xr.DataArray(dims=['frequency'], coords={'frequency':frequencies},
                         attrs={'name':'mr',
                                'long_name':'real part of the refractive index',
                                'units':'dimensionless',
                                'comment':'Computed using Iwabuchi 2010 model as implemented in the snowScatt code (https://github.com/OPTIMICe-team/snowScatt) for a temperature of 263K'})
ref_idx_i = xr.DataArray(dims=['frequency'], coords={'frequency':frequencies},
                         attrs={'name':'mi',
                                'long_name':'imaginary part of the refractive index',
                                'units':'dimensionless',
                                'comment':'Computed using Iwabuchi 2010 model as implemented in the snowScatt code (https://github.com/OPTIMICe-team/snowScatt) for a temperature of 263K'})
##############################################################
# Microphysical and shape-related variables
##############################################################
masses = xr.DataArray(dims=['Dmax'], coords={'Dmax':sizes},
                      attrs={'name':'ma',
                             'long_name':'mass',
                             'units':'kilograms',
                             'comment':'ice density {} kg/m3'.format(ice_density)})
aspect = xr.DataArray(dims=['Dmax'], coords={'Dmax':sizes},
                      attrs={'name':'ar',
                             'long_name':'aspect_ratio',
                             'units':'dimensionless'})
dipole_resolution = xr.DataArray(dims=['Dmax'], coords={'Dmax':sizes},
                                 attrs={'name':'d',
                                        'long_name':'dipole_resolution',
                                        'units':'meters'})
volume_eq_size_parameter = xr.DataArray(dims=['Dmax'], coords={'Dmax':sizes},
                                        attrs={'name':'xeq',
                                               'long_name':'volume_equivalent_size_parameter',
                                               'units':'dimensionless'})
#shapefile_path = xr.DataArray(dims=['Dmax'], coords={'Dmax':sizes},
#                                    attrs={'name':'shape_path',
#                                           'long_name':'path to the shapefile connected to the scattering calculations',
#                                           'comment':'this is not read from the logfile, it is just the shapefile in the expected location. It allows to debug the data collection rather than the scattering calculations'})


###########################################################
# Define variables in the database
###########################################################

# These are X,Y,Z cartesian coordinates that define the propagation direction in DDA calculation (Particle Reference Frame)
# Also the Ey and Ex polarization direction as they are suggested by ADDA

X = xr.DataArray(dims=['direction'], coords={'direction':angles},
                 attrs={'name':'prop_x',
                        'long_name':'x cartesian propagation direction'})
Y = xr.DataArray(dims=['direction'], coords={'direction':angles},
                 attrs={'name':'prop_y',
                        'long_name':'y cartesian propagation direction'})
Z = xr.DataArray(dims=['direction'], coords={'direction':angles},
                 attrs={'name':'prop_z',
                        'long_name':'z cartesian propagation direction'})

EyX = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'EyX',
                          'long_name':'x cartesian Ey parallel polarization component'})
EyY = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'EyY',
                          'long_name':'y cartesian Ey parallel polarization component'})
EyZ = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'EyZ',
                          'long_name':'z cartesian Ey parallel polarization component'})

ExX = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'ExX',
                          'long_name':'x cartesian Ex perpendicular polarization component'})
ExY = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'ExY',
                          'long_name':'y cartesian Ex perpendicular polarization component'})
ExZ = xr.DataArray(dims=['direction'], coords={'direction':angles},
                   attrs={'name':'ExZ',
                          'long_name':'z cartesian Ex perpendicular polarization component'})

# These are the actual output of the DDA calculations, S amplitude matrices, separated in forward and backward scattering, real and imaginary part
# Now I am also recording the angle index to simplify debugging

# Angle indexes
a_group = xr.DataArray(dims=dims, coords=coords,
                       attrs={'long_name':'Number of subdivisions in the icosahedral grid'})
a_idx = xr.DataArray(dims=dims, coords=coords,
                     attrs={'long_name':'Index of the angle node'})

# Backward components
S1br = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S1 element of the amplitude matrix backward',
                           'units':'meters'})
S2br = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S2 element of the amplitude matrix backward',
                           'units':'meters'})
S3br = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S3 element of the amplitude matrix backward',
                           'units':'meters'})
S4br = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S4 element of the amplitude matrix backward',
                           'units':'meters'})
S1bi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S1 element of the amplitude matrix backward',
                           'units':'meters'})
S2bi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S2 element of the amplitude matrix backward',
                           'units':'meters'})
S3bi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S3 element of the amplitude matrix backward',
                           'units':'meters'})
S4bi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S4 element of the amplitude matrix backward',
                           'units':'meters'})

# Forward components
S1fr = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S1 element of the amplitude matrix forward',
                           'units':'meters'})
S2fr = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S2 element of the amplitude matrix forward',
                           'units':'meters'})
S3fr = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S3 element of the amplitude matrix forward',
                           'units':'meters'})
S4fr = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part S4 element of the amplitude matrix forward',
                           'units':'meters'})
S1fi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S1 element of the amplitude matrix forward',
                           'units':'meters'})
S2fi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S2 element of the amplitude matrix forward',
                           'units':'meters'})
S3fi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S3 element of the amplitude matrix forward',
                           'units':'meters'})
S4fi = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part S4 element of the amplitude matrix forward',
                           'units':'meters'})

# Important functions (parsers)

def parse_command(command):
    lam = float(command.split('-lambda')[-1].split()[0])
    m1 = float(command.split('-m')[-1].split()[0])
    m2 = float(command.split('-m')[-1].split()[1])
    return lam, complex(m1, m2)

findstr = [ 'lambda:',
            'Incident propagation vector:',
            'Incident polarization Y(par):',
            'Incident polarization X(per):']
def parse_log(logfile):
    with open(logfile, 'r') as f:
        results = {}
        for target in findstr:
            line = f.readline()
            while (not line.startswith(target)):
                line = f.readline()
                if not line:
                    break
            results[target] = line
    prop = results['Incident propagation vector:'].split('(')[-1].split(')')[0].split(',')
    x, y, z = prop
    parY = results['Incident polarization Y(par):'].split('(')[-1].split(')')[0].split(',')
    Yx, Yy, Yz = parY
    perX = results['Incident polarization X(per):'].split('(')[-1].split(')')[0].split(',')
    Xx, Xy, Xz = perX
    return float(x), float(y), float(z), float(Yx), float(Yy), float(Yz), float(Xx), float(Xy), float(Xz), float(results['lambda:'].split()[-1])

##########################################################################################
# Loop 1 just to take x,y,z
##########################################################################################
for fld in shpflds[0:1]: # The assumption is that only one shape is needed to get all directions
    #print(fld)
    oriflds=sorted(glob('{}{}*/'.format(fld, freqstr))) # TODO probably fine but check that it catches all directions
    #print(len(oriflds))
    for iori, ori in enumerate(oriflds):
        logfile = ori + 'log'
        x, y, z, Yx, Yy, Yz, Xx, Xy, Xz, lam = parse_log(logfile)
        X.loc[iori] = x
        Y.loc[iori] = y
        Z.loc[iori] = z
        EyX.loc[iori] = Yx
        EyY.loc[iori] = Yy
        EyZ.loc[iori] = Yz
        ExX.loc[iori] = Xx
        ExY.loc[iori] = Xy
        ExZ.loc[iori] = Xz

        #print(ori, x, y, z)

k = 2.0*np.pi/lam # calculate just one time here the wavenumber
wavelengths.loc[freq] = lam # derive the wavelength

###################################################################################################
# Extract the microphysical and scattering properties in the PRF for every file and every direction
###################################################################################################
for i,fld in enumerate(shpflds):
    print("{} of {} {}".format(i, len(shpflds), fld))
    ###################mass and aspect ratio calculation#################        
    dipoles = np.loadtxt(fld+'shapefile.dat')
    oriflds=sorted(glob('{}{}*/'.format(fld, freqstr))) # TODO probably fine but check that it catches all directions
    #shapefile_path[i] = fld

    #### Get microphysics from one log ####
    logfile = oriflds[0] + 'log'
    findstr = ['refractive index:',
               'Dipole size:',
               'Volume-equivalent size parameter:']
    with open(logfile, 'r') as f:
        results = {}
        for target in findstr:
            line = f.readline()
            while (not line.startswith(target)):
                line = f.readline()
                if not line:
                    break
            results[target] = line
        mstr = results['refractive index:']
        ref_index = complex(mstr.split(':')[-1].split('i')[0]+'j')
        dip_resolution = float(results['Dipole size:'].split(':')[-1].split('(')[0])
        x_eff = float(results['Volume-equivalent size parameter:'].split(':')[-1].split('\n')[0])
        ref_idx_r.loc[freq] = ref_index.real
        ref_idx_i.loc[freq] = ref_index.imag
        dipole_resolution[i] = dip_resolution
        volume_eq_size_parameter[i] = x_eff
    masses[i] = ice_density*len(dipoles)*dipole_resolution[i]**3
    Z_extension = (np.max(dipoles[:,2]) - np.min(dipoles[:,2]))*dipole_resolution[i] # meters
    aspect[i] = Z_extension*1.0e3/Dmax[i]

    for iori, ori in enumerate(oriflds):
        Astr, Nstr = ori.split('/')[-2].split('_')
        a_group.loc[Dmax[i], freq, iori] = int(Astr) # TODO comment me if you do not have a icosahedral grid
        a_idx.loc[Dmax[i], freq, iori] = int(Nstr)   # TODO comment me if you do not have a icosahedral grid

        #logfile = ori + 'log' # No need for the log here
        amplfile = ori + 'ampl'
        #x, y, z, lam = parse_log(logfile) # k is going to be always the same for now #k = 2.0*np.pi/lam
        ampl = pd.read_csv(amplfile, sep=' ', index_col='theta',
                           header=0, names=col_ampl)
        ampl['s1'] = ampl.s1r + 1.0j*ampl.s1i
        ampl['s2'] = ampl.s2r + 1.0j*ampl.s2i
        ampl['s3'] = ampl.s3r + 1.0j*ampl.s3i
        ampl['s4'] = ampl.s4r + 1.0j*ampl.s4i
        ampl.drop(col_ampl[1:], axis=1, inplace=True)
        s1, s2, s3, s4 = (ampl.loc[180.0]*1.0j/k).values # Convert Bohren-Huffman to Mischenko
        S1br.loc[Dmax[i], freq, iori] = s1.real
        S2br.loc[Dmax[i], freq, iori] = s2.real
        S3br.loc[Dmax[i], freq, iori] = s3.real
        S4br.loc[Dmax[i], freq, iori] = s4.real
        S1bi.loc[Dmax[i], freq, iori] = s1.imag
        S2bi.loc[Dmax[i], freq, iori] = s2.imag
        S3bi.loc[Dmax[i], freq, iori] = s3.imag
        S4bi.loc[Dmax[i], freq, iori] = s4.imag
        
        s1, s2, s3, s4 = (ampl.loc[0.0]*1.0j/k).values # Convert Bohren-Huffman to Mischenko
        S1fr.loc[Dmax[i], freq, iori] = s1.real
        S2fr.loc[Dmax[i], freq, iori] = s2.real
        S3fr.loc[Dmax[i], freq, iori] = s3.real
        S4fr.loc[Dmax[i], freq, iori] = s4.real
        S1fi.loc[Dmax[i], freq, iori] = s1.imag
        S2fi.loc[Dmax[i], freq, iori] = s2.imag
        S3fi.loc[Dmax[i], freq, iori] = s3.imag
        S4fi.loc[Dmax[i], freq, iori] = s4.imag


## Finalize dataset and write netCDF file
variables = {'S1br':S1br,
             'S2br':S2br,
             'S3br':S3br,
             'S4br':S4br,
             'S1bi':S1bi,
             'S2bi':S2bi,
             'S3bi':S3bi,
             'S4bi':S4bi,
             'S1fr':S1fr,
             'S2fr':S2fr,
             'S3fr':S3fr,
             'S4fr':S4fr,
             'S1fi':S1fi,
             'S2fi':S2fi,
             'S3fi':S3fi,
             'S4fi':S4fi,
             'X':X,
             'Y':Y,
             'Z':Z,
             'EyX':EyX,
             'EyY':EyY,
             'EyZ':EyZ,
             'ExX':ExX,
             'ExY':ExY,
             'ExZ':ExZ,
             'wavelength':wavelengths,
             'mass':masses,
             'ar':aspect,
             'dipole_resolution':dipole_resolution,
             'xeff':volume_eq_size_parameter,
             'mr':ref_idx_r,
             'mi':ref_idx_i,
             'a_group':a_group, # TODO comment me if you do not have a icosahedral grid
             'a_idx':a_idx      # TODO comment me if you do not have a icosahedral grid
             #'shape_fld':shapefile_path
            }

global_attributes = {'created_by':os.environ['USER'],
                     'host_machine':socket.gethostname(),
                     'particle_properties':'icon 2 mom ice crystals',
                     'particle_name':'{}'.format(parttype),
                     'path':'{}'.format(basename),
                     'created_on':str(datetime.now()),
                     'comment':'Particle properties are read from the first logfile of the many directions, hopefully it did not change'}

dataset = xr.Dataset(data_vars=variables,
                     coords=coords,
                     attrs=global_attributes)

encoding = {k:{'zlib': True, 'complevel': 9} for k in variables}

dataset.to_netcdf('./{}{}.nc'.format(output_name, freqstr[:-1]), encoding=encoding)

#dataset.to_netcdf('/work/bb1314/ncdata/agg_lambda_5e-5to0.001/scatter_data_2momplateden_upto12.91mm_lambda_5e-5to0.001_Cband.nc'.format(basename), encoding=encoding)

#old_data = xr.open_dataset('/work/bb1314/ncdata/microphysical_data_plates.nc')
#new_data = xr.merge([old_data, dataset])
#new_data.to_netcdf('/work/bb1314/ncdata/scatt_and_micro_data_plate_X.nc'.format(basename), encoding=encoding)
