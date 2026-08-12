#!/usr/bin/env python
# coding: utf-8

# In[1]:


## IMPORTS

import numpy as np
import matplotlib as pyplot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import glob
import re
from collections import defaultdict
from scipy.optimize import least_squares


# In[2]:


## CONSTANTS (Can replace later with astropy dict for a cleaner look and keeping track of units)
G = 4.30091e-6 #in consistent units with galaxy params


# In[3]:


## FUNCTIONS

def burkertprofile (r,rho,rc): #Burkert profile for scipy
    x = r/rc
    M=2*(np.pi*rho*rc**3)*(np.log(1+x)+0.5*np.log(1+x**2)-np.arctan(x))
    return np.sqrt(G*M/r)


def nfwprofile (r,rho, rs): #NFW profile for scipy
    x=r/rs
    M= (4*np.pi*rho*rs**3*(np.log(1+x)-x/(1+x)))

    return np.sqrt(G*M/r)


def isothermal (r,rho,rc): #Isothermal profile for scipy
    innerterm = 1-(rc/r)*np.arctan(r/rc)
    outerterm = 4*np.pi*G*rho*rc**2*innerterm
    return np.sqrt(outerterm)

def uniform(): #Uniform profile for scipy
    return

def totalvelo(r,rho,rc,Vgas,Vdisk,Vbul,halofunc): #Calculating total velocity curve
    Vhalo = halofunc(r,rho,rc)
    return np.sqrt(Vgas**2+Vdisk**2+Vhalo**2+Vbul**2)


def residuals(params, galaxy, halofunc):
    rho0, rc = params

    model = totalvelo(
        galaxy["R"],
        rho0,
        rc,
        galaxy["Vgas"],
        galaxy["Vdisk"],
        galaxy["Vbul"],
        halofunc
    )

    return (model - galaxy["Vobs"]) / galaxy["Err"]

def fitgalaxy(galaxy, halofunc, p0):
    result = least_squares(residuals,x0=p0,args=(galaxy, halofunc),bounds=(0, np.inf))

    return result
def reducedchi(result, galaxy):
    return np.sum(result.fun**2)/(len(galaxy["R"])-2)

def fitstatistics(galaxy, Vmodel, n_params=2):
    Vobs = galaxy["Vobs"]
    Err = galaxy["Err"]
    residuals = Vobs - Vmodel
    normalized = residuals / Err
    N = len(Vobs)
    chi2 = np.sum(normalized**2)
    chi2red = chi2 / (N - n_params)
    overlap = np.sum(np.abs(residuals) <= Err) / N * 100

    return chi2red, overlap

def parameter_uncertainties(result):
    J = result.jac
    dof = len(result.fun) - len(result.x)
    chi2_red = np.sum(result.fun**2) / dof
    cov = np.linalg.pinv(J.T @ J) * chi2_red
    uncertainties = np.sqrt(np.diag(cov))
    return uncertainties


# In[4]:


## LOADING IN GALAXY DATA
galaxyfiles = sorted(glob.glob("*_rotmod.dat")) ## ALL GALAXY FILES
datalist = []
galaxies = []

for file in galaxyfiles:
    data = np.loadtxt(file, unpack=True)
    R=data[0]
    Vobs=data[1]
    Err=data[2]
    Vgas=data[3]
    Vdisk=data[4]
    Vbul=data[5]
    SBdisk=data[6]
    SBbul=data[7]

    galaxies.append({"name": file.replace("_rotmod.dat", ""),"R": R,"Vobs": Vobs,"Err": Err,"Vgas": Vgas,
        "Vdisk": Vdisk,"Vbul": Vbul,"SBdisk": SBdisk,"SBbul": SBbul
    })
spirals = [
    "NGC7331",
    "NGC2903",
    "NGC3198",
    "NGC3521",
    "NGC5055",
    "NGC2403",
    "NGC6503",
    "NGC4559",
    "NGC5585",
    "NGC1003",
    "UGC03546",
    "UGC00128"
]

dwarves = [
    "DDO154",
    "NGC3741",
    "DDO161",
    "IC2574",
    "UGC05721",
    "UGCA442"
]

lsb = [
    "NGC6503",
    "NGC5585",
    "NGC1003",
    "UGC00128"
]

dm_dominated = [
    "DDO154",
    "NGC6503",
    "NGC5585"
]


# In[5]:


# General rotation curves not fit to


fig, axes = plt.subplots(6, 3, figsize=(18, 20))
axes = axes.flatten()

for ax, galaxy in zip(axes, galaxies[:18]):

    ax.errorbar(galaxy["R"],galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2)
    ax.set_title(galaxy["name"], fontsize=10)
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("V (km/s)")
    ax.grid(alpha=0.3)

plt.suptitle(rf"SPARC Galaxy Rotation Curves", size = 25, fontweight = 'bold', y=1)
plt.tight_layout()
plt.show()


# In[6]:


fig, axes = plt.subplots(6, 3, figsize=(18, 20))
axes = axes.flatten()

for ax, galaxy in zip(axes, galaxies[:18]): 
    resultburkert = fitgalaxy(galaxy, burkertprofile,p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile,p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal,p0=(1e7, 5))

    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x
    sigma_rhob, sigma_rcb = parameter_uncertainties(resultburkert)
    sigma_rhon, sigma_rcn = parameter_uncertainties(resultnfw)
    sigma_rhoi, sigma_rci = parameter_uncertainties(resultiso)

    R = galaxy["R"]

    Vmodelburkert = totalvelo(R,rhob,rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)
    Vmodelnfw = totalvelo(R,rhon,rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)
    Vmodelisothermal = totalvelo(R,rhoi,rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    ax.errorbar(R,galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2, label = 'Data')

    ax.plot(R,Vmodelburkert,color="xkcd:cerulean",linewidth=3, alpha = 0.8, label=rf"Burkert") #for the sake of practice, I'd like to play around with making sure
    ax.plot(R,Vmodelnfw,color="xkcd:goldenrod",linewidth=2, alpha = 0.8, label = rf"NFW")
    ax.plot(R,Vmodelisothermal,color="xkcd:raspberry",linewidth=2, alpha = 0.8, label = rf"Isothermal")

    ax.set_title(galaxy["name"], fontsize=10)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(alpha=0.3)

    if ax is axes[0]:
        ax.legend()

plt.suptitle(rf"Fits to SPARC Galaxy Rotation Curves", size = 25, fontweight = 'bold', y=1)

plt.tight_layout()

plt.show()




# In[7]:


fig, axes = plt.subplots(6, 3, figsize=(18, 20))
axes = axes.flatten()

for ax, galaxy in zip(axes, galaxies[:18]):

    # Fit each model
    resultburkert = fitgalaxy(galaxy, burkertprofile,p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile,p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal,p0=(1e7, 5))
    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]

    # Model velocities
    Vmodelburkert = totalvelo(
        R, rhob, rcb,
        galaxy["Vgas"],
        galaxy["Vdisk"],
        galaxy["Vbul"],
        burkertprofile
    )

    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)

    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    # Residuals
    residualburkert = galaxy["Vobs"] - Vmodelburkert
    residualnfw = galaxy["Vobs"] - Vmodelnfw
    residualiso = galaxy["Vobs"] - Vmodelisothermal


    chib, overlapb = fitstatistics(galaxy, Vmodelburkert)

    chin, overlapn = fitstatistics(galaxy, Vmodelnfw)

    chii, overlapi = fitstatistics(galaxy, Vmodelisothermal)


    text = (rf"Burkert: $\chi_\nu^2={chib:.2f}$, {overlapb:.1f}%""\n"rf"NFW: $\chi_\nu^2={chin:.2f}$, {overlapn:.1f}%""\n"rf"Iso: $\chi_\nu^2={chii:.2f}$, {overlapi:.1f}%")

    ax.text(0.03,0.97,text,
    transform=ax.transAxes,fontsize=7,verticalalignment="top",bbox=dict(facecolor="white",alpha=0.8))

    ax.errorbar(R,residualburkert,yerr=galaxy["Err"],fmt=".",color="xkcd:cerulean",markersize=4, alpha = 0.8,label="Burkert")
    ax.errorbar(R,residualnfw,yerr=galaxy["Err"],fmt=".",color="xkcd:goldenrod",markersize=4, alpha = 0.8,label="NFW")
    ax.errorbar(R,residualiso,yerr=galaxy["Err"],fmt=".",color="xkcd:raspberry",markersize=4, alpha = 0.8,label="Isothermal")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(galaxy["name"], fontsize=10)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel(r"$V_{\rm obs}-V_{\rm model}$ (km/s)")
    ax.grid(alpha=0.3)

    if ax is axes[0]:
        ax.legend()

plt.suptitle(rf"Rotation Curve Residuals", size = 25, fontweight = 'bold', y=1)
plt.tight_layout()
plt.show()


# In[8]:


fig, axes = plt.subplots(6, 3, figsize=(18, 20))
axes = axes.flatten()

for ax, galaxy in zip(axes, galaxies[:18]):

    resultburkert = fitgalaxy(galaxy, burkertprofile,p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile,p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal,p0=(1e7, 5))

    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]

    Vmodelburkert = totalvelo(R, rhob, rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)
    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)
    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    residualburkert = (galaxy["Vobs"] - Vmodelburkert) / galaxy["Err"]
    residualnfw = (galaxy["Vobs"] - Vmodelnfw) / galaxy["Err"]
    residualiso = (galaxy["Vobs"] - Vmodelisothermal) / galaxy["Err"]

    ax.errorbar(R,residualburkert,fmt=".",color="xkcd:plum",label="Burkert")
    ax.errorbar(R,residualnfw,fmt=".",color="xkcd:goldenrod",label="NFW")
    ax.errorbar(R,residualiso,fmt=".",color="xkcd:reddish",label="Isothermal")

    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel(r"$(V_{\rm obs}-V_{\rm model})/\sigma_V$")

    ax.set_title(galaxy["name"], fontsize=10)
    ax.set_xlabel("Radius (kpc)")
    ax.grid(alpha=0.3)
    if ax is axes[0]:
        ax.legend()

plt.suptitle(rf"Normalized Residuals", size = 25, fontweight = 'bold', y=1)
plt.tight_layout()
plt.show()




# In[ ]:





# In[9]:


dm_names = [
    "DDO154",
    "NGC6503",
    "NGC5585"
]

dm_galaxies = [
    galaxy for galaxy in galaxies
    if galaxy["name"] in dm_names
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for ax, galaxy in zip(axes, dm_galaxies):

    resultburkert = fitgalaxy(galaxy, burkertprofile, p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile, p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal, p0=(1e7, 5))

    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]

    Vmodelburkert = totalvelo(R, rhob, rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)
    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)
    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    ax.errorbar(R,galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2)

    ax.plot(R, Vmodelburkert,color="xkcd:cerulean",linewidth=3,alpha=0.8,label="Burkert")
    ax.plot(R, Vmodelnfw,color="xkcd:goldenrod",linewidth=2,alpha=0.8,label="NFW")
    ax.plot(R, Vmodelisothermal,color="xkcd:raspberry",linewidth=2,alpha=0.8,label="Isothermal")
    ax.set_title(galaxy["name"], fontstyle = 'italic', fontsize=11)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(alpha=0.3)

axes[0].legend()

plt.suptitle(
    "Fits to Dark Matter Dominated Galaxy Rotation Curves",
    fontsize=20,
    fontweight="bold", y=1
)

plt.tight_layout()
plt.show()


# In[10]:


lsb_names = [
    "NGC6503",
    "NGC5585",
    "NGC1003",
    "UGC00128"
]

lsb_galaxies = [
    galaxy for galaxy in galaxies
    if galaxy["name"] in lsb_names
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, galaxy in zip(axes, lsb_galaxies):

    resultburkert = fitgalaxy(galaxy, burkertprofile, p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile, p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal, p0=(1e7, 5))

    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]

    Vmodelburkert = totalvelo(R, rhob, rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)
    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)
    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    ax.errorbar(R,galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2)

    ax.plot(R, Vmodelburkert,color="xkcd:cerulean",linewidth=3,alpha=0.8,label="Burkert")
    ax.plot(R, Vmodelnfw,color="xkcd:goldenrod",linewidth=2,alpha=0.8,label="NFW")
    ax.plot(R, Vmodelisothermal,color="xkcd:raspberry",linewidth=2,alpha=0.8,label="Isothermal")

    ax.set_title(galaxy["name"], fontstyle = 'italic', fontsize=11)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(alpha=0.3)

axes[0].legend()

plt.suptitle("Fits to Low Surface Brightness Galaxy Rotation Curves",fontsize=15,fontweight="bold", y =1)

plt.tight_layout()
plt.show()


# In[11]:


dwarf_names = [
    "DDO154",
    "NGC3741",
    "DDO161",
    "IC2574",
    "UGC05721",
    "UGCA442"
]

dwarves = [
    galaxy for galaxy in galaxies
    if galaxy["name"] in dwarf_names
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, galaxy in zip(axes, dwarves):

    resultburkert = fitgalaxy(galaxy, burkertprofile, p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile, p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal, p0=(1e7, 5))

    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]

    Vmodelburkert = totalvelo(R, rhob, rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)

    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)

    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    ax.errorbar(R,galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2)

    ax.plot(R, Vmodelburkert,color="xkcd:cerulean",linewidth=3,alpha=0.8,label="Burkert")
    ax.plot(R, Vmodelnfw,color="xkcd:goldenrod",linewidth=2,alpha=0.8,label="NFW")
    ax.plot(R, Vmodelisothermal,color="xkcd:raspberry",linewidth=2,alpha=0.8,label="Isothermal")

    ax.set_title(galaxy["name"], fontstyle ='italic', fontsize=11)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(alpha=0.3)

axes[0].legend()

plt.suptitle("Fits to Dwarf Galaxy Rotation Curves",fontsize=17,fontweight="bold", y=1)

plt.tight_layout()
plt.show()


# In[12]:


spiral_names = [
    "NGC7331",
    "NGC2903",
    "NGC3198",
    "NGC3521",
    "NGC5055",
    "NGC2403",
    "NGC6503",
    "NGC4559",
    "NGC5585",
    "NGC1003",
    "UGC03546",
    "UGC00128"
]

spirals = [
    galaxy for galaxy in galaxies
    if galaxy["name"] in spiral_names
]

fig, axes = plt.subplots(4, 3, figsize=(15, 18))
axes = axes.flatten()

for ax, galaxy in zip(axes, spirals):
    resultburkert = fitgalaxy(galaxy, burkertprofile, p0=(1e7, 5))
    resultnfw = fitgalaxy(galaxy, nfwprofile, p0=(1e6, 10))
    resultiso = fitgalaxy(galaxy, isothermal, p0=(1e7, 5))


    rhob, rcb = resultburkert.x
    rhon, rsn = resultnfw.x
    rhoi, rci = resultiso.x

    R = galaxy["R"]


    Vmodelburkert = totalvelo(R, rhob, rcb,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],burkertprofile)
    Vmodelnfw = totalvelo(R, rhon, rsn,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],nfwprofile)
    Vmodelisothermal = totalvelo(R, rhoi, rci,galaxy["Vgas"],galaxy["Vdisk"],galaxy["Vbul"],isothermal)

    ax.errorbar(R,galaxy["Vobs"],yerr=galaxy["Err"],fmt="k.",markersize=3,capsize=2)

    ax.plot(R, Vmodelburkert,color="xkcd:cerulean",linewidth=3,alpha=0.8,label="Burkert")
    ax.plot(R, Vmodelnfw,color="xkcd:goldenrod",linewidth=2,alpha=0.8,label="NFW")
    ax.plot(R, Vmodelisothermal,color="xkcd:raspberry",linewidth=2,alpha=0.8,label="Isothermal")
    ax.set_title(galaxy["name"], fontstyle = 'italic',  fontsize=11)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
axes[0].legend()
plt.suptitle("Fits to Spiral Galaxy Rotation Curves",fontsize=17,fontweight="bold", y=1)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




