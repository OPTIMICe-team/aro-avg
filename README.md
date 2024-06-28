# aro-avg
Routines to make Azimuthally Random Orientations (ARO) averaging of scattering properties (amplitude matrix) from DDA measurements

## Scripts
Run first either collect_SSP_data_serial.py or the parallel version of it to produce an netCDF database of scattering properties

Run create_aro_DB.py to perform ARO averages for all particles and frequencies in the SSP database (this is only parallel version)

## Libraries
The algebra to calculate vectors and angles in rotated reference frames is defined in geometry3D.py
The algebra to derive transformation matrices in Laboratory and Particle Reference Frames (LRF and PRF) is written in common.py

Why is it called common.py? It was part of a previous repository that compared DDA and T-matrix. I had a dda.py library a tmm.py and of course these transformation matrices were common to both methods so... you got it

