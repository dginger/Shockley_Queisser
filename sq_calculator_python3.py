# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:52:45 2016

@author: mziffer  Mark Ziffer, University of Washington
updated to python 3.6 7/20/18 David S. Ginger, University of Washington, dginger@uw.edu
Based on the code of Steve Byrnes (http://sjbyrnes.com/sq.html), but much faster using spline integration

Reference: Shockley, W. & Queisser, H. J. J. Appl. Phys. 32, 510, doi:10.1063/1.1736034 (1961)
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from scipy.integrate import simps, quad
from scipy.interpolate import splrep, splint
from scipy.optimize import fmin



h = constants.physical_constants['Planck constant'][0] # units of J*s
h_ev = constants.physical_constants['Planck constant in eV s'][0]
c_nm = (constants.physical_constants['speed of light in vacuum'][0]) * 1e9
c = (constants.physical_constants['speed of light in vacuum'][0])
  
e_charge = constants.physical_constants['elementary charge'][0] 
kb_ev = constants.physical_constants['Boltzmann constant in eV/K'][0]

"""User settings"""
#print ('started')   #just a debug line
Tcell = 298 #temperature of solar cell in degrees K
bandgap = 0.5 #enter bandgap in eV

print ('Tcell = %.3f' %(Tcell))

plot_jv = 'n' #'y' if you want to plot the SQ JV curve for "bandgap"
              #'n' if not (faster)


plot_bandgap_array = 'n' #'y' if you want to plot SQ parameters for an array of bandgaps 
                         #    starting from "mbandgap_array_min" to "bandgap_array_max" 
                         #    with number of points "num_points_bandgap_array"
                         #    (see below)
                         
                         #'n' if you just want SQ data for one bandgap (faster)

bandgap_array_min = 0.5 #in eV
bandgap_array_max = 3 # in eV
num_points_bandgap_array = 27


"""Programming below"""
bandgap_array = np.linspace(bandgap_array_min, bandgap_array_max, num_points_bandgap_array)
#First convert AM1.5 spectrum from W/m^2/nm to W/m^2/ev

astmg173 = np.loadtxt(r'ASTMG173.csv', delimiter = ',', skiprows = 2)

am15_wav = np.copy(astmg173[:,0]) #AM1.5 wavelength axis in nm

am15 = np.copy(astmg173[:,2]) #AM1.5 in units of W/m^2/nm = J/s*m^2/nm

total_power_nm = simps(am15, x = am15_wav) #Integrate over nm to check that total power density = 1000 W/m^2


am15_ev = h_ev * (c_nm) / (am15_wav )
am15_wats_ev = am15 * (h_ev * c_nm/ ((am15_ev) ** 2.0))

am15_ev_flip = am15_ev[::-1] 
am15_wats_ev_flip = am15_wats_ev[::-1]


total_power_ev = simps(am15_wats_ev_flip, x = am15_ev_flip) #Integrate over eV to check that total power density = 1000 W/m^2


am15_photons_ev  = am15_wats_ev_flip / (am15_ev_flip * e_charge)

am15_photons_nm = am15 / (am15_ev * e_charge)

total_photonflux_ev = simps(am15_photons_ev, x = am15_ev_flip)


total_photonflux_nm = simps(am15_photons_nm , x = am15_wav)


total_photonflux_ev_splrep = splrep(am15_ev_flip, am15_photons_ev)

emin = am15_ev_flip[0]
emax = am15_ev_flip[len(am15_ev_flip) - 1]

def solar_photons_above_gap(Egap): #units of photons / sec *m^2
    return splint(Egap, emax,total_photonflux_ev_splrep) 
    

def RR0(Egap):
    integrand = lambda eV : eV ** 2.0 / (np.exp(eV / (kb_ev * Tcell)) - 1)
    integral = quad(integrand, Egap, emax, full_output=1)[0]
    return ((2.0 * np.pi / ((c ** 2.0) * (h_ev ** 3.0)))) * integral
        
def current_density(V, Egap): #to get from units of amps / m^2 to mA/ cm^2 ---multiply by 1000 to convert to mA ---- multiply by (0.01 ^2) to convert to cm^2
    cur_dens =  e_charge * (solar_photons_above_gap(Egap) - RR0(Egap) * np.exp( V / (kb_ev * Tcell)))    
    return cur_dens * 1000 * (0.01 ** 2.0)
def JSC(Egap): 
    return current_density(0, Egap) 
    
def VOC(Egap):
    return (kb_ev * Tcell) * np.log(solar_photons_above_gap(Egap) / RR0(Egap))

    
def fmax(func_to_maximize, initial_guess=0):
    """return the x that maximizes func_to_maximize(x)"""
    func_to_minimize = lambda x : -func_to_maximize(x)
    return fmin(func_to_minimize, initial_guess, disp=False)[0]    

def V_mpp_Jmpp_maxpower_maxeff_ff(Egap):

    vmpp = fmax(lambda V : V * current_density(V, Egap))    
    jmpp = current_density(vmpp, Egap)
    
    maxpower =  vmpp * jmpp
    max_eff = maxpower / (total_power_ev * 1000 * (0.01 ** 2.0))
    jsc_return =  JSC(Egap)
    voc_return = VOC(Egap)
    ff = maxpower / (jsc_return * voc_return)    
    return [vmpp, jmpp, maxpower, max_eff, ff, jsc_return, voc_return]


maxpcemeta = V_mpp_Jmpp_maxpower_maxeff_ff(bandgap)

print ('For Bandgap = %.3f eV, TCell = %.3f K:\nJSC = %.3f mA/cm^2\nVOC = %.3f V\nFF = %.3f\nPCE = %.3f' % (bandgap, Tcell, maxpcemeta[5], maxpcemeta[6],maxpcemeta[4], maxpcemeta[3] * 100))

if plot_bandgap_array == 'y':
    
    pce_array = np.empty_like(bandgap_array)
    ff_array = np.empty_like(bandgap_array)
    voc_array = np.empty_like(bandgap_array)
    jsc_array = np.empty_like(bandgap_array)
    for i in range(len(bandgap_array)):
        metadata = V_mpp_Jmpp_maxpower_maxeff_ff(bandgap_array[i])
        pce_array[i] = metadata[3] 
        ff_array[i] = metadata[4]
        voc_array[i] = metadata[6]
        jsc_array[i] = metadata[5]
        
    plt.figure()
    plt.title('Tcell = %.3f K' %(Tcell))
    plt.xlim(bandgap_array[0], bandgap_array[len(bandgap_array) - 1])
    plt.ylabel('PCE (%)')
    plt.xlabel('Bandgap (eV)')
    plt.plot(bandgap_array, pce_array * 100)
    
    plt.figure()
    plt.title('Tcell = %.3f K' %(Tcell))
    plt.ylim(0, 1)
    plt.xlim(bandgap_array[0], bandgap_array[len(bandgap_array) - 1])
    plt.ylabel('Fill Factor')
    plt.xlabel('Bandgap (eV)')
    plt.plot(bandgap_array, ff_array)
    
    plt.figure()
    plt.title('Tcell = %.3f K' %(Tcell))
    plt.xlim(bandgap_array[0], bandgap_array[len(bandgap_array) - 1])
    plt.ylabel('Jsc (mA/cm^2)')
    plt.xlabel('Bandgap (eV)')
    plt.plot(bandgap_array, jsc_array)
    
    plt.figure()
    plt.title('Tcell = %.3f K' %(Tcell))
    plt.xlim(bandgap_array[0], bandgap_array[len(bandgap_array) - 1])
    plt.ylabel('Voc (V)')
    plt.xlabel('Bandgap (eV)')
    plt.plot(bandgap_array, voc_array, label = 'S-Q Voc')
    plt.plot(bandgap_array, bandgap_array, '--', label = 'Bandgap')
    plt.legend(loc = 'best')
    
    print ('--')

else:
    print ('--')


def JV_curve(Egap):
    volt_array = np.linspace(0, VOC(Egap), 200)
    j_array = np.empty_like(volt_array)
    for i in range(len(volt_array)):
        j_array[i] = current_density(volt_array[i], Egap)
    return [volt_array, j_array]


if plot_jv == 'y':
    jv_meta = JV_curve(bandgap)
    v_array = jv_meta[0]
    jv_array = jv_meta[1]

    plt.figure()
    plt.ylabel('Current Density (mA/cm^2)')
    plt.xlabel('Voltage (V)')
    plt.plot(v_array, -jv_array)
    print ('--')
else:
    print ('--')
