#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:48:15 2023

@author: dori
"""

import numpy as np

def eulerZYZ(alpha, beta, gamma):
    a = np.deg2rad(alpha)
    b = np.deg2rad(beta)
    c = np.deg2rad(gamma)
    s1 = np.sin(a)
    c1 = np.cos(a)
    s2 = np.sin(b)
    c2 = np.cos(b)
    s3 = np.sin(c)
    c3 = np.cos(c)
    
    R = np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                  [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                  [-c3*s2, s2*s3, c2]])
    return R


def surface3d(ax, x, y, z, alpha=0.2, size=5):
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    point = np.array([0.0, 0.0, 0.0]) # just in case I will want a shifted plane someday
    d = -point.dot(np.array([x, y, z]))
    
    if z:
        xx, yy = np.meshgrid(np.arange(size) - 0.5*(size-1), np.arange(size) - 0.5*(size-1))
        zz = -(x*xx + y*yy + d)/z
    else: # if z==0 the plane goes through z axis
        xx, zz = np.meshgrid(np.arange(size) - 0.5*(size-1), np.arange(size) - 0.5*(size-1))
        yy = -(x*xx + d)/y
    ax.plot_surface(xx, yy, zz, alpha=alpha)
        

def arrow3d(ax, length=2.5, width=0.01, head=0.1, headwidth=5,
                theta=0, phi=0, offset=(0,0,0), **kw):
    w = width
    h = head
    hw = headwidth
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    a = [[0, 0], [w, 0], [w, (1-h)*length], [hw*w,(1-h)*length],[0,length]]
    a = np.array(a)

    r, th = np.meshgrid(a[:, 0], np.linspace(0, 2*np.pi, 30))
    z = np.tile(a[:,1], r.shape[0]).reshape(r.shape)
    x = r*np.sin(th)
    y = r*np.cos(th)

    rot_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    rot_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                      [np.sin(phi), np.cos(phi), 0],
                      [0, 0, 1]])

    b1 = np.dot(rot_y, np.c_[x.flatten(),y.flatten(),z.flatten()].T)
    b2 = np.dot(rot_z, b1)
    b2 = b2.T+np.array(offset)
    x = b2[:, 0].reshape(r.shape); 
    y = b2[:, 1].reshape(r.shape); 
    z = b2[:, 2].reshape(r.shape); 
    ax.plot_surface(x, y, z, **kw)

    
def cart2pol(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    th = np.arccos(z/r)
    ph = np.arctan2(y, x)
    ph = ph if ph >= 0.0 else 2.0*np.pi + ph
    return np.rad2deg(th), np.rad2deg(ph), r


def pol2cart(th, ph, r=1.0):
    th = np.deg2rad(th)
    ph = np.deg2rad(ph)
    z = r*np.cos(th)
    re = r*np.sin(th)
    x = re*np.cos(ph)
    y = re*np.sin(ph)
    return x, y, z