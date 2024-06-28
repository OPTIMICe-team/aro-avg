#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:50:36 2024

@author: dori

This script takes a SSP database created with collect_*.py and for each 
particle in the SSP computes the ARO averaged amplitude matrices in the forward
and backward direction. The backward direction is saved in the form of the real
4x4 scattering matrix for easier computations of radar quantities.

It is very compute intensive, but not much memory intensive so it is worth
spawning a lot of parallel processes (as long as you have many particles in the
SSP database). This script only parallelize for distinct particles, but if you
merged multiple frequencies in the SSP database it will loop through them

ALL ANGLES ARE IN DEGREES !!!
"""

from time import time
import concurrent.futures

import os
import socket
from datetime import datetime
import numpy as np
import xarray as xr

from common import dB, lab2partRF, find_backward, S2Z
from geometry3D import pol2cart

MaxWorkers = 256 # max number of cores
out_comment = 'This is just a test'  # this will be printed int the output global attributes
out_name = 'scat_plate_aro_horiz.nc' # name of the output .nc file

###############################################################################
# Set the range of alpha and beta angles, exploit particle periodicity on gamma
Ngamma = 25
Nalpha = 50
gammas = np.linspace(0.0, 60.0, Ngamma, endpoint=False)  # TODO set gamma periodicity, if end=60deg it is a 6-fold periodic axial symmetry like hexagonal prisms
alphas = np.linspace(0.0, 360.0, Nalpha, endpoint=False)

##############################################################################
# Set the range of elevations of the radar and canting angles beta
elevs = np.arange(0.0, 0.1, 10.0)
bets =  np.arange(0.0, 90.1, 1.0)

###############################################################################
# Gimme a fullpath to a valid SSP database
ssd = xr.open_dataset('./plate005.6GHz.nc')


##############################################################################
# HERE WE GO !!!!!!!!!!
#############################################################################
ssd.load() # speed up access?
grid = np.vstack([ssd.X.values, ssd.Y.values, ssd.Z.values])
Dmax = ssd.Dmax
frequencies = ssd.frequency

elevations = xr.IndexVariable(dims='elevation', data=elevs,
                              attrs={'long_name':'radar elevation angle',
                                     'units':'degrees'})

betas = xr.IndexVariable(dims='beta', data=bets,
                         attrs={'long_name':'Beta Euler angle',
                                'units':'degrees'})

dims = ['Dmax', 'frequency', 'elevation', 'beta']
coords = {'Dmax':Dmax, 'frequency':frequencies,
          'elevation':elevations, 'beta':betas}

Zdr = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'differential reflectivity',
                          'units':'dB'})

Z11 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z11 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z12 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z12 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z13 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z13 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z14 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z14 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z21 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z21 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z22 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z22 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z23 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z23 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z24 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z24 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z31 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z31 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z32 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z32 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z33 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z33 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z34 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z34 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z41 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z41 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z42 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z42 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z43 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z43 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

Z44 = xr.DataArray(dims=dims, coords=coords,
                   attrs={'long_name':'Z44 element of the backward scattering matrix',
                          'units':'millimeteres**2'})

S11r = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part of the S11 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S11i = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part of the S11 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S12r = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part of the S12 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S12i = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part of the S12 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S21r = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part of the S21 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S21i = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part of the S21 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S22r = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'real part of the S22 element of the forward amplitude matrix',
                           'units':'millimeteres'})

S22i = xr.DataArray(dims=dims, coords=coords,
                    attrs={'long_name':'imaginary part of the S22 element of the forward amplitude matrix',
                           'units':'millimeteres'})


###############################################################################
# Starts mapping the sampling directions in the SSP database

idxs = np.zeros((Ngamma, Nalpha, len(elevations), len(betas)), dtype=int)
start = time()
for iel, el in enumerate(elevs[0:]):
    for ibeta, beta in enumerate(betas[0:]):
        ph0L = 0.0
        th0L = 90.0 - el
        th1L, ph1L = find_backward(th0L, ph0L)
        for igam, gam in enumerate(gammas):
            for ialp, alp in enumerate(alphas):
                L2P = lab2partRF(th0L, th1L, ph0L, ph1L, alp, beta, gam)
                th0P, th1P, ph0P, ph1P, Rho, invRho = L2P
                x0, y0, z0 = pol2cart(th0P, ph0P)
                dots = grid.T.dot(np.array([x0, y0, z0]))
                idxs[igam, ialp, iel, ibeta] = np.argmax(dots)
end = time()
print('Initial evaluation {} seconds'.format(end-start))
freq = frequencies.values[0]


###############################################################################
# Define the averaging function

def _average_one_particle(ds, ssd):
    Dx = ds.Dmax
    SD = ssd.sel(Dmax=Dx, frequency=freq)
    SD.load() # faster?
    #print(Dx)
    
    for iel, el in enumerate(elevs[0:]):
        for ibeta, beta in enumerate(betas[0:]):
            #print(el, beta)
            ph0L = 0.0
            th0L = 90.0 - el
            th1L, ph1L = find_backward(th0L, ph0L)
            
            ZLt = np.zeros((4, 4))
            SLt = np.zeros((2, 2))+0.0j*np.zeros((2, 2))
            for igam, gam in enumerate(gammas):
                for ialp, alp in enumerate(alphas):
                    L2P = lab2partRF(th0L, th1L, ph0L, ph1L, alp, beta, gam)
                    th0P, th1P, ph0P, ph1P, Rho, invRho = L2P
                    #x0, y0, z0 = pol2cart(th0P, ph0P)
                    #dots = grid.T.dot(np.array([x0,y0,z0]))
                    #idx = np.argmax(dots) # this can be done in advance!!!
                    idx = idxs[igam, ialp, iel, ibeta]
                    Sd = SD.isel(direction=idx)
                    S1 = (Sd.S1br + 1.0j*Sd.S1bi).values
                    S2 = (Sd.S2br + 1.0j*Sd.S2bi).values
                    S3 = (Sd.S3br + 1.0j*Sd.S3bi).values
                    S4 = (Sd.S4br + 1.0j*Sd.S4bi).values
                    SP = np.array([[S1, S4], [S3, S2]])
                    SL = invRho.dot(SP).dot(Rho)
                    ZL = S2Z(*SL.flatten())
                    
                    ZLt += ZL/(Ngamma*Nalpha)
                    
                    # Do forward
                    L2P = lab2partRF(th0L, th0L, ph0L, ph0L, alp, beta, gam)
                    th0P, th1P, ph0P, ph1P, Rho, invRho = L2P
                    S1f = (Sd.S1fr + 1.0j*Sd.S1fi).values
                    S2f = (Sd.S2fr + 1.0j*Sd.S2fi).values
                    S3f = (Sd.S3fr + 1.0j*Sd.S3fi).values
                    S4f = (Sd.S4fr + 1.0j*Sd.S4fi).values
                    SPf = np.array([[S1, S4], [S3, S2]])
                    SLf = invRho.dot(SPf).dot(Rho)
                    SLt += SLf/(Ngamma*Nalpha)
                    HH = 2*np.pi*(ZLt[0,0] - ZLt[0,1] - ZLt[1,0] + ZLt[1,1])
                    VV = 2*np.pi*(ZLt[0,0] + ZLt[0,1] + ZLt[1,0] + ZLt[1,1])
                    #print('Zdr = dda {0:3.3f}'.format(dB(HH/VV)))
                    ds.Zdr.loc[freq, el, beta] = dB(HH/VV)
                    ds.Z11.loc[freq, el, beta] = ZLt[0, 0]
                    ds.Z12.loc[freq, el, beta] = ZLt[0, 1]
                    ds.Z13.loc[freq, el, beta] = ZLt[0, 2]
                    ds.Z14.loc[freq, el, beta] = ZLt[0, 3]
                    ds.Z21.loc[freq, el, beta] = ZLt[1, 0]
                    ds.Z22.loc[freq, el, beta] = ZLt[1, 1]
                    ds.Z23.loc[freq, el, beta] = ZLt[1, 2]
                    ds.Z24.loc[freq, el, beta] = ZLt[1, 3]
                    ds.Z31.loc[freq, el, beta] = ZLt[2, 0]
                    ds.Z32.loc[freq, el, beta] = ZLt[2, 1]
                    ds.Z33.loc[freq, el, beta] = ZLt[2, 2]
                    ds.Z34.loc[freq, el, beta] = ZLt[2, 3]
                    ds.Z41.loc[freq, el, beta] = ZLt[3, 0]
                    ds.Z42.loc[freq, el, beta] = ZLt[3, 1]
                    ds.Z43.loc[freq, el, beta] = ZLt[3, 2]
                    ds.Z44.loc[freq, el, beta] = ZLt[3, 3]
                    
                    ds.S11r.loc[freq, el, beta] = SLt[0, 0].real
                    ds.S11i.loc[freq, el, beta] = SLt[0, 0].imag
                    ds.S12r.loc[freq, el, beta] = SLt[0, 1].real
                    ds.S12i.loc[freq, el, beta] = SLt[0, 1].imag
                    ds.S21r.loc[freq, el, beta] = SLt[1, 0].real
                    ds.S21i.loc[freq, el, beta] = SLt[1, 0].imag
                    ds.S22r.loc[freq, el, beta] = SLt[1, 1].real
                    ds.S22i.loc[freq, el, beta] = SLt[1, 1].imag
    return ds # not needed in serial, the function works, in parallel it creates copies so the internal assignment does not work

## Finalize dataset and write netCDF file
variables = {'Zdr':Zdr,
             'Z11':Z11,
             'Z12':Z12,
             'Z13':Z13,
             'Z14':Z14,
             'Z21':Z21,
             'Z22':Z22,
             'Z23':Z23,
             'Z24':Z24,
             'Z31':Z31,
             'Z32':Z32,
             'Z33':Z33,
             'Z34':Z34,
             'Z41':Z41,
             'Z42':Z42,
             'Z43':Z43,
             'Z44':Z44,
             'S11r':S11r,
             'S11i':S11i,
             'S12r':S12r,
             'S12i':S12i,
             'S21r':S21r,
             'S21i':S21i,
             'S22r':S22r,
             'S22i':S22i}

global_attributes = {'created_by':os.environ['USER'],
                     'host_machine':socket.gethostname(),
                     'particle_properties':'icon 2mom cloud ice',
                     'created_on':str(datetime.now()),
                     'comment':out_comment}

dataset = xr.Dataset(data_vars=variables,
                     coords=coords,
                     attrs=global_attributes)

###############################################################################
# Finally, compute averages, collect data from parallel jobs and save results
###############################################################################

init = time()
future = []
with concurrent.futures.ProcessPoolExecutor(max_workers=MaxWorkers) as executor:
    for Dx in Dmax[0:]:
        print(Dx.values)
        future.append(executor.submit(_average_one_particle, ds=dataset.sel(Dmax=Dx), ssd=ssd))
print("multithreads done in {} seconds".format(time()-init))
init = time()
results = [i.result() for i in future]
for r in results:
    dataset.loc[dict(Dmax=r.Dmax)] = r
print("Assignment of results done in {} seconds".format(time()-init))

encoding = {k:{'zlib': True, 'complevel': 9} for k in variables}
dataset.to_netcdf(out_name, encoding=encoding)
