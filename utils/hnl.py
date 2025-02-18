"""
# ==================================================================
#   Python module
#
#   This module provides methods to compute the lifetime and
#   branching ratio of HNLs given its mass and couplings as
#   input parameters.
#
#   Created: 30/11/2014 Elena Graverini (elena.graverini@cern.ch)
#
#   Updated: 07/10/2017 Kyrylo Bondarenko (bondarenko@lorentz.leidenuniv.nl)
#
#   Sample usage:
#     ipython -i hnl.py
#     In [1]: b = HNL(1.,[1.e-8, 2.e-8, 1.e-9],True)
#     HNLbranchings instance initialized with couplings:
#          U2e   = 1e-08
#      U2mu  = 2e-08
#      U2tau = 1e-09
#     and mass:
#      m = 1.0 GeV
#     In [2]: b.computeNLifetime()
#     Out[2]: 4.777721453160521e-05
#     In [3]: b.findBranchingRatio('N -> pi mu')
#     Out[3]: 0.11826749348890987
#
# ==================================================================
"""

from __future__ import print_function

import math
import sys, os
import numpy as np
import scipy.interpolate
import scipy.integrate
import shipunit as u
import hnlbr.data.particle as dat

# Load PDG database
pdata = dat.ParticleData()

def PDGname(particle):
    """
    Change particle name for use with the PDG database
    """
    if '1' in particle:
        particle = particle.replace('1',"'")
    if particle == 'nu': return 'nu_e' # simple workaround to handle 3nu channel
    return particle

quark_masses = {
    'u': 2.2e-3,
    'd': 4.7e-3,
    's': 96e-3,
    'c': 1.27,
    'b': 4.18,
    't': 173.21
}

def mass(particle):
    """
    Read particle mass from PDG database
    """
    if particle in quark_masses:
        return quark_masses[particle]
    particle = PDGname(particle)
    pdg_id = pdata.pdg_id(particle)
    return pdata.mass(pdg_id)

class CKMmatrix():
    """
    CKM matrix, from http://pdg.lbl.gov/2017/reviews/rpp2016-rev-ckm-matrix.pdf
    """
    def __init__(self):
        self.Vud = 0.9743
        self.Vus = 0.2251
        self.Vub = 3.6e-03
        self.Vcd = 0.2249
        self.Vcs = 0.9735
        self.Vcb = 4.11e-02
        self.Vtd = 8.6e-03
        self.Vts = 4.0e-02
        self.Vtb = 0.999

class constants():
    """
    Store some constants useful for HNL physics
    """
    def __init__(self):
        self.decayConstant = {'pi+':0.130*u.GeV,
                            'pi0':0.130*u.GeV,
                            'eta':1.2*0.130*u.GeV,
                            'eta1':0.152*u.GeV,
                            'eta_c':0.335*u.GeV,
                            'rho+':0.209*u.GeV,
                            'rho0':0.209*u.GeV,
                            'omega':0.195*u.GeV,
                            'phi':0.229*u.GeV,
                            'D_s+':0.249*u.GeV,
                            'D*_s+':0.315*u.GeV} # decay constants f_h of pseudoscalar and vector mesons
        #self.GF = 1.166379e-05/(u.GeV*u.GeV) # Fermi's constant (GeV^-2)
        self.GF = 1.16638E-05/(u.GeV*u.GeV) # Fermi's constant (GeV^-2) <-- Oleg's G_F 
        self.s2thetaw = 0.23126 # square sine of the Weinberg angle
        self.heV = 6.58211928*pow(10.,-16) # no units or it messes up!!
        self.hGeV = self.heV * pow(10.,-9) # no units or it messes up!!

# Load some useful constants
c = constants()

class HNLbranchings():
    """
    Lifetime and total and partial decay widths of an HNL
    """
    def __init__(self, mass, couplings, debug=False):
        """
        Initialize with mass and couplings of the HNL

        Inputs:
        mass (GeV)
        couplings (list of [U2e, U2mu, U2tau])
        """
        self.U2 = couplings
        self.U = [math.sqrt(ui) for ui in self.U2]
        self.MN = mass*u.GeV
        self.CKM = CKMmatrix()
        self.CKMelemSq = {'pi+':self.CKM.Vud**2.,
                        'rho+':self.CKM.Vud**2.,
                        'D_s+':self.CKM.Vcs**2.,
                        'D*_s+':self.CKM.Vcs**2.,
                        'K+':self.CKM.Vus**2.,
                        # Quarks (up,down)
                        # up: 1=u, 2=c, 3=t
                        # down: 1=d, 2=s, 3=b
                        (1,1):self.CKM.Vud**2., (1,2):self.CKM.Vus**2., (1,3):self.CKM.Vub**2.,
                        (2,1):self.CKM.Vcd**2., (2,2):self.CKM.Vcs**2., (2,3):self.CKM.Vcb**2.,
                        (3,1):self.CKM.Vtd**2., (3,2):self.CKM.Vts**2., (3,3):self.CKM.Vtb**2.}
        self.decays = [ 'N -> nu nu nu',
                        'N -> e- e+ nu_e',
                        'N -> e- e+ nu_mu',
                        'N -> e- e+ nu_tau',
                        'N -> e- mu+ nu_mu',
                        'N -> mu- e+ nu_e',
                        'N -> pi0 nu_e',
                        'N -> pi0 nu_mu',
                        'N -> pi0 nu_tau',
                        'N -> pi+ e-',
                        'N -> mu- mu+ nu_e',
                        'N -> mu- mu+ nu_mu',
                        'N -> mu- mu+ nu_tau',
                        'N -> pi+ mu-',
                        'N -> eta nu_e',
                        'N -> eta nu_mu',
                        'N -> eta nu_tau',
                        'N -> rho0 nu_e',
                        'N -> rho0 nu_mu',
                        'N -> rho0 nu_tau',
                        'N -> rho+ e-',
                        'N -> omega nu_e',
                        'N -> omega nu_mu',
                        'N -> omega nu_tau',
                        'N -> rho+ mu-',
                        'N -> eta1 nu_e',
                        'N -> eta1 nu_mu',
                        'N -> eta1 nu_tau',
                        'N -> phi nu_e',
                        'N -> phi nu_mu',
                        'N -> phi nu_tau',
                        'N -> e- tau+ nu_tau',
                        'N -> tau- e+ nu_e',
                        'N -> mu- tau+ nu_tau',
                        'N -> tau- mu+ nu_mu',
                        'N -> D_s+ e-',
                        'N -> D_s+ mu-',
                        'N -> D*_s+ e-',
                        'N -> D*_s+ mu-',
                        'N -> eta_c nu_e',
                        'N -> eta_c nu_mu',
                        'N -> eta_c nu_tau' ]
        if self.MN>=1.:
            self.QCD_corr = self.QCD_correction()

        if debug:
            print("HNLbranchings instance initialized with couplings:")
            print("\tU2e   = %s"%self.U2[0])
            print("\tU2mu  = %s"%self.U2[1])
            print("\tU2tau = %s"%self.U2[2])
            print("and mass:")
            print("\tm = %s GeV"%(self.MN))

    def sqrt_lambda(self,a,b,c):
        """
        Useful function for decay kinematics. Returns 0 for kinematically forbidden region
        """
        l = a**2 + b**2 + c**2 - 2*a*b - 2*b*c - 2*a*c
        if l<0:
            return 0
        else:
            return math.sqrt(l)

    def QCD_correction(self):
        """
        Returns 3-loops QCD correction to HNL decay width into quarks
        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        arr = np.loadtxt(os.path.join(this_dir, 'hnlbr/alpha_s.dat'))
        E = arr[:,0]
        alpha_s = arr[:,1]
        interpolator = scipy.interpolate.interp1d(E, alpha_s, kind='linear')
        a_s = interpolator(self.MN)
        qcd_corr = a_s / math.pi
        qcd_corr += 5.2 * (a_s / math.pi)**2.
        qcd_corr += 26.4 * (a_s / math.pi)**3.
        return qcd_corr

    def Width_3nu(self):
        """
        Returns the HNL decay width into three neutrinos
        """
        width = (c.GF**2.)*(self.MN**5.)*sum(self.U2)/(192.*(u.pi**3.))
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Width_nu_f_fbar(self, alpha, beta):
        """
        Returns the HNL tree level decay width into the fermion-antifermion pair and a neutrino (decay through Z boson or Z and W)

        Inputs:
        - alpha is a flavour of neutrino (1,2 or 3), determine the mixing angle U_alpha
        - beta is a flavour of the fermions: (1, 2 or 3) for charge leptons or (4, 5, 6, 7, 8, 9) for quarks
        """
        if (alpha not in [1,2,3]) or (beta not in [1,2,3,4,5,6,7,8,9]):
            print('Width_nu_f_fbar ERROR: unknown channel alpha =',alpha,' beta =',beta)
            quit()
        l = [None,'e-','mu-','tau-','u','d','s','c','b','t']
        x = mass(l[beta]) / self.MN
        if x > 0.5: # the decay is kinematically forbidden
            return 0
        width = (c.GF**2.)*(self.MN**5.)*self.U2[alpha-1]/(192.*(math.pi**3.))
        L = 0.
        if x>0.01:
            logContent = (1. - 3.*x**2. - (1.-x**2.)*math.sqrt(1. - 4.*x**2.) ) / ( (x**2.)*(1 + math.sqrt(1. - 4.*x**2.)) )
        else:
            logContent = x**4 + 4.*x**6 + 14.*x**8
        if logContent > 0:
            L = math.log( logContent )
        if beta < 4: # lepton decay
            NZ = 1
            if alpha == beta: # interference case
                C1 = 0.25*(1. + 4.*c.s2thetaw + 8.*c.s2thetaw**2.)
                C2 = 0.5*c.s2thetaw*(2.*c.s2thetaw +1.)
            else: # no interference
                C1 = 0.25*(1. - 4.*c.s2thetaw + 8.*c.s2thetaw**2.)
                C2 = 0.5*c.s2thetaw*(2.*c.s2thetaw -1.)
        else: # decay into quarks
            NZ = 3
            if (beta in [4,7,9]): # up quarks
                C1 = 0.25*(1. - 8./3.*c.s2thetaw + 32./9.*c.s2thetaw**2.)
                C2 = 1./3.*c.s2thetaw*(4./3.*c.s2thetaw - 1.)
            if (beta in [5,6,8]): # down quarks
                C1 = 0.25*(1. - 4./3.*c.s2thetaw + 8./9.*c.s2thetaw**2.)
                C2 = 1./6.*c.s2thetaw*(2./3.*c.s2thetaw - 1.)
        width = width*NZ*( C1*( (1.-14.*x**2. -2.*x**4. -12.*x**6.)*math.sqrt(1-4.*x**2) +12.*x**4. *(-1.+x**4.)*L )
                        + 4.*C2*( x**2. *(2.+10.*x**2. -12.*x**4.) * math.sqrt(1.-4.*x**2) + 6.*x**4. *(1.-2.*x**2+2.*x**4)*L ) )
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Integrand(self,xx,xi):
        """
        Function to integrate needed for numerical integration using ROOT. Needed for 3-body decays trough W boson.
        First argument is a list of 1 number, argument. Second is the list of 3 numbers xi = mi/MN
        """
        x = xx[0]
        xl2 = xi[0]**2
        xu2 = xi[1]**2
        xd2 = xi[2]**2
        if x==0: # Workaround for division by zero
            return 0
        res = 1./x
        res *= (x - xl2 - xd2)
        res *= (1. + xu2 - x)
        res *= self.sqrt_lambda(x,xl2,xd2)
        res *= self.sqrt_lambda(1.,x,xu2)
        return res

    def I(self,x1,x2,x3):
        """
        Numerical integral needed for 3-body decays trough W boson.
        xi = mi/MN
        """
        xmin = (x1 + x3)**2
        xmax = (1. - x2)**2
        val, err = scipy.integrate.quad(
            lambda xx: self.Integrand([xx], [x1,x2,x3]),
            xmin, xmax
        )
        res = 12. * val
        return res

    def Width_l1_l2_nu2(self, alpha, beta):
        """
        Returns the HNL decay width into two different flavour leptons and a neutrino (decay through W boson)

        Inputs:
        - alpha is a flavour of the first lepton (1,2 or 3), determine the mixing angle U_alpha
        - beta is a flavour of the second lepton and neutrino (1, 2 or 3)
        """
        if (alpha not in [1,2,3]) or (beta not in [1,2,3]):
            print('Width_l1_l2_nu2 ERROR: unknown channel alpha =',alpha,' beta =',beta)
            quit()
        if alpha==beta: # The interference case is handled by Width_nu_f_fbar function, workaround for a total width calculation
            return 0
        l = [None,'e-','mu-','tau-']
        x1 = mass(l[alpha])/self.MN
        x2 = mass(l[beta])/self.MN
        if x1+x2>1: # The decay is kinematically forbidden
            return 0
        width = (c.GF**2.)*(self.MN**5.)*self.U2[alpha-1]/(192.*(math.pi**3.))
        width = width*self.I(x1,x2,0)
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Width_l_u_d(self, alpha, beta, gamma):
        """
        Returns the HNL tree level decay width into a charged lepton, up quark and down quark (decay through W boson)

        Inputs:
        - alpha is a flavour of the charged lepton (1,2 or 3), determine the mixing angle U_alpha
        - beta is the up quark generation (1, 2, 3 for u, c, t)
        - gamma is the down quark generation (1, 2, 3 for d, s, b)
        """
        if (alpha not in [1,2,3]) or (beta not in [1,2,3]) or (gamma not in [1,2,3]):
            print('Width_l_u_d ERROR: unknown channel alpha =',alpha,' beta =',beta,' gamma =',gamma)
            quit()
        l = [None,'e-','mu-','tau-']
        u = [None,'u','c','t']
        d = [None,'d','s','b']
        xl = mass(l[alpha])/self.MN
        xu = mass(u[beta])/self.MN
        xd = mass(d[gamma])/self.MN
        if xl+xu+xd>1: # The decay is kinematically forbidden
            return 0
        width = (c.GF**2.)*(self.MN**5.)*self.U2[alpha-1]/(192.*(math.pi**3.))
        width *= 3*self.CKMelemSq[(beta, gamma)]
        width *= self.I(xl,xu,xd)
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Width_H0_nu(self, H, alpha):
        """
        Returns the HNL decay width into neutral meson and neutrino

        Inputs:
        - H is a string (name of the meson)
        - alpha is a flavour of the nu (1, 2 or 3), determine the mixing angle U_alpha
        """
        if (H not in ['pi0','eta','rho0','omega','eta1','phi','eta_c']) or (alpha not in [1,2,3]):
            print('Width_H0_nu ERROR: unknown channel H0 =',H,' alpha =',alpha)
            quit()
        x = mass(H)/self.MN
        if x > 1: # The decay is kinematically forbidden
            return 0.
        width = (c.GF**2.)*(c.decayConstant[H]**2.)*(self.MN**3.)*self.U2[alpha-1]/(32.*math.pi)
        if H in ['pi0','eta','eta1','eta_c']: # pseudoscalar mesons
            width *= (1 - x**2.)**2.
        if H in ['rho0','omega','phi']: # vector mesons
            width *= (1. + 2*x**2.)*(1. - x**2.)**2.
            if H=='rho0':
                width *= (1. - 2.*c.s2thetaw)**2.
            if H=='omega':
                width *= (16.*c.s2thetaw**2.)/9.
            if H=='phi':
                width *= (1. - 4./3.*c.s2thetaw)**2.
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Width_H_l(self, H, alpha):
        """
        Returns the HNL decay width into charged meson and lepton

        Inputs:
        - H is a string (name of the positively charged meson)
        - alpha is the lepton flavour (1, 2 or 3), determine the mixing angle U_alpha
        """
        if (H not in ['pi+','rho+','D_s+','D*_s+']) or (alpha not in [1,2,3]):
            print('Width_H_l ERROR: unknown channel H =',H,' alpha =',alpha)
            quit()
        l = [None,'e-','mu-','tau-']
        xl = mass(l[alpha])/self.MN
        xh = mass(H)/self.MN
        if xl+xh > 1: # The decay is kinematically forbidden
            return 0.
        width = (c.GF**2.)*(c.decayConstant[H]**2.)*self.CKMelemSq[H]*(self.MN**3.)*self.U2[alpha-1]/(16.*math.pi)
        width *= self.sqrt_lambda(1., xl**2, xh**2)
        if H in ['pi+','D_s+']: # pseudoscalar mesons
            width *= ( (1. - xl**2.)**2. - xh**2. *(1. + xl**2.) )
        if H in ['rho+','D*_s+']: # vector mesons
            width *= ( (1. - xl**2.)**2. + xh**2.*(1. + xl**2.) - 2.*xh**4. )
        width = 2.*width # Majorana case (charge conjugate channels)
        return width

    def Width_charged_leptons(self):
        """
        Returns the total HNL leptonic decay width with charged leptons in final state
        """
        width = 0.
        for l1 in [1,2,3]:
            for l2 in [1,2,3]:
                width += self.Width_nu_f_fbar(l1,l2)
                width += self.Width_l1_l2_nu2(l1,l2)
        return width

    def Width_neutral_mesons(self):
        """
        Returns the total HNL decay width into a neutral meson and a neutrino
        """
        mesons = ['pi0','eta','rho0','omega','eta1','phi','eta_c']
        width = 0.
        for m in mesons:
            for l in [1,2,3]:
                width += self.Width_H0_nu(m,l)
        return width

    def Width_charged_mesons(self):
        """
        Returns the total HNL decay width into a charged meson and a charged lepton
        """
        mesons = ['pi+','rho+','D_s+','D*_s+']
        width = 0.
        for m in mesons:
            for l in [1,2,3]:
                width += self.Width_H_l(m,l)
        return width

    def Width_quarks_neutrino(self):
        """
        Returns the total HNL decay width with quarks and neutrino in final state. Uses 3-loops alpha_s correction
        """
        if self.MN<1.: # decay into quarks is not applicable at low HNL mass
            return 0
        width = 0.
        for l in [1,2,3]:
            for q in [4,5,6,7,8,9]:
                width += self.Width_nu_f_fbar(l,q)
        width *= (1. + self.QCD_corr)
        return width

    def Width_quarks_lepton(self):
        """
        Returns the total HNL decay width with quarks and charged lepton in final state. Uses 3-loops alpha_s correction
        """
        if self.MN<1.: # decay into quarks is not applicable at low HNL mass
            return 0
        width = 0.
        for l in [1,2,3]:
            for u in [1,2,3]:
                for d in [1,2,3]:
                    width += self.Width_l_u_d(l,u,d)
        width *= (1. + self.QCD_corr)
        return width

    def NDecayWidth(self):
        """
        Returns the total HNL decay width
        """
        leptonWidth = self.Width_3nu() + self.Width_charged_leptons()
        mesonWidth = self.Width_neutral_mesons() + self.Width_charged_mesons()
        quarkWidth = self.Width_quarks_neutrino() + self.Width_quarks_lepton()
        totalWidth = leptonWidth + max([mesonWidth, quarkWidth])
        return totalWidth

    def findBranchingRatio(self, decayString):
        """
        Returns the branching ratio of the selected HNL decay channel

        Inputs:
        - decayString is a string describing the decay, in the form 'N -> stuff1 ... stuffN'
        """
        br = 0.
        totalWidth = self.NDecayWidth()

        if (decayString not in self.decays) and (decayString not in ['N -> hadrons','N -> charged hadrons']):
            print('findBranchingRatio ERROR: unknown decay %s'%decayString)
            quit()

        if decayString == 'N -> nu nu nu' or decayString == 'N -> 3nu': br = self.Width_3nu() / totalWidth # inclusive
        if decayString == 'N -> e- e+ nu_e': br = self.Width_nu_f_fbar(1,1) / totalWidth
        if decayString == 'N -> e- e+ nu_mu': br = self.Width_nu_f_fbar(2,1) / totalWidth
        if decayString == 'N -> e- e+ nu_tau': br = self.Width_nu_f_fbar(3,1) / totalWidth
        if decayString == 'N -> e- mu+ nu_mu': br = self.Width_l1_l2_nu2(1,2) / totalWidth
        if decayString == 'N -> mu- e+ nu_e': br = self.Width_l1_l2_nu2(2,1) / totalWidth
        if decayString == 'N -> pi0 nu_e': br = self.Width_H0_nu('pi0',1) / totalWidth
        if decayString == 'N -> pi0 nu_mu': br = self.Width_H0_nu('pi0',2) / totalWidth
        if decayString == 'N -> pi0 nu_tau': br = self.Width_H0_nu('pi0',3) / totalWidth
        if decayString == 'N -> pi+ e-': br = self.Width_H_l('pi+',1) / totalWidth
        if decayString == 'N -> mu- mu+ nu_e': br = self.Width_nu_f_fbar(1,2) / totalWidth
        if decayString == 'N -> mu- mu+ nu_mu': br = self.Width_nu_f_fbar(2,2) / totalWidth
        if decayString == 'N -> mu- mu+ nu_tau': br = self.Width_nu_f_fbar(3,2) / totalWidth
        if decayString == 'N -> pi+ mu-': br = self.Width_H_l('pi+',2) / totalWidth
        if decayString == 'N -> eta nu_e': br = self.Width_H0_nu('eta',1) / totalWidth
        if decayString == 'N -> eta nu_mu': br = self.Width_H0_nu('eta',2) / totalWidth
        if decayString == 'N -> eta nu_tau': br = self.Width_H0_nu('eta',3) / totalWidth
        if decayString == 'N -> rho0 nu_e': br = self.Width_H0_nu('rho0',1) / totalWidth
        if decayString == 'N -> rho0 nu_mu': br = self.Width_H0_nu('rho0',2) / totalWidth
        if decayString == 'N -> rho0 nu_tau': br = self.Width_H0_nu('rho0',3) / totalWidth
        if decayString == 'N -> rho+ e-': br = self.Width_H_l('rho+',1) / totalWidth
        if decayString == 'N -> omega nu_e': br = self.Width_H0_nu('omega',1) / totalWidth
        if decayString == 'N -> omega nu_mu': br = self.Width_H0_nu('omega',2) / totalWidth
        if decayString == 'N -> omega nu_tau': br = self.Width_H0_nu('omega',3) / totalWidth
        if decayString == 'N -> rho+ mu-': br = self.Width_H_l('rho+',2) / totalWidth
        if decayString == 'N -> eta1 nu_e': br = self.Width_H0_nu('eta1',1) / totalWidth
        if decayString == 'N -> eta1 nu_mu': br = self.Width_H0_nu('eta1',2) / totalWidth
        if decayString == 'N -> eta1 nu_tau': br = self.Width_H0_nu('eta1',3) / totalWidth
        if decayString == 'N -> phi nu_e': br = self.Width_H0_nu('phi',1) / totalWidth
        if decayString == 'N -> phi nu_mu': br = self.Width_H0_nu('phi',2) / totalWidth
        if decayString == 'N -> phi nu_tau': br = self.Width_H0_nu('phi',3) / totalWidth
        if decayString == 'N -> e- tau+ nu_tau': br = self.Width_l1_l2_nu2(1,3) / totalWidth
        if decayString == 'N -> tau- e+ nu_e': br = self.Width_l1_l2_nu2(3,1) / totalWidth
        if decayString == 'N -> mu- tau+ nu_tau': br = self.Width_l1_l2_nu2(2,3) / totalWidth
        if decayString == 'N -> tau- mu+ nu_mu': br = self.Width_l1_l2_nu2(3,2) / totalWidth
        if decayString == 'N -> D_s+ e-': br = self.Width_H_l('D_s+',1) / totalWidth
        if decayString == 'N -> D_s+ mu-': br = self.Width_H_l('D_s+',2) / totalWidth
        if decayString == 'N -> D*_s+ e-': br = self.Width_H_l('D*_s+',1) / totalWidth
        if decayString == 'N -> D*_s+ mu-': br = self.Width_H_l('D*_s+',2) / totalWidth
        if decayString == 'N -> eta_c nu_e': br = self.Width_H0_nu('eta_c',1) / totalWidth
        if decayString == 'N -> eta_c nu_mu': br = self.Width_H0_nu('eta_c',2) / totalWidth
        if decayString == 'N -> eta_c nu_tau': br = self.Width_H0_nu('eta_c',3) / totalWidth

        if decayString == 'N -> hadrons':
            mesonWidth = self.Width_neutral_mesons() + self.Width_charged_mesons()
            quarkWidth = self.Width_quarks_neutrino() + self.Width_quarks_lepton()
            br = max([mesonWidth, quarkWidth]) / totalWidth
        if decayString == 'N -> charged hadrons':
            br = max([self.Width_charged_mesons(), self.Width_quarks_lepton()]) / totalWidth
        return br

    def allowedChannels(self):
        """
        Returns a dictionary of kinematically allowed decay channels

        Inputs:
        - decayString is a string describing the decay, in the form 'N -> stuff1 ... stuffN'
        """
        m = self.MN
        allowedDecays = {'N -> nu nu nu':'yes'}
        if m > 2.*mass('e-'):
            allowedDecays.update({'N -> e- e+ nu_e':'yes'})
            allowedDecays.update({'N -> e- e+ nu_mu':'yes'})
            allowedDecays.update({'N -> e- e+ nu_tau':'yes'})
        if m > mass('e-') + mass('mu-'):
            allowedDecays.update({'N -> e- mu+ nu_mu':'yes'})
            allowedDecays.update({'N -> mu- e+ nu_e':'yes'})
        if m > mass('pi0'):
            allowedDecays.update({'N -> pi0 nu_e':'yes'})
            allowedDecays.update({'N -> pi0 nu_mu':'yes'})
            allowedDecays.update({'N -> pi0 nu_tau':'yes'})
        if m > mass('pi+') + mass('e-'):
            allowedDecays.update({'N -> pi+ e-':'yes'})
        if m > 2.*mass('mu-'):
            allowedDecays.update({'N -> mu- mu+ nu_e':'yes'})
            allowedDecays.update({'N -> mu- mu+ nu_mu':'yes'})
            allowedDecays.update({'N -> mu- mu+ nu_tau':'yes'})
        if m > mass('pi+') + mass('mu-'):
            allowedDecays.update({'N -> pi+ mu-':'yes'})
        if m > mass('eta'):
            allowedDecays.update({'N -> eta nu_e':'yes'})
            allowedDecays.update({'N -> eta nu_mu':'yes'})
            allowedDecays.update({'N -> eta nu_tau':'yes'})
        if m > mass('rho0'):
            allowedDecays.update({'N -> rho0 nu_e':'yes'})
            allowedDecays.update({'N -> rho0 nu_mu':'yes'})
            allowedDecays.update({'N -> rho0 nu_tau':'yes'})
        if m > mass('rho+') + mass('e-'):
            allowedDecays.update({'N -> rho+ e-':'yes'})
        if m > mass('omega'):
            allowedDecays.update({'N -> omega nu_e':'yes'})
            allowedDecays.update({'N -> omega nu_mu':'yes'})
            allowedDecays.update({'N -> omega nu_tau':'yes'})
        if m > mass('rho+') + mass('mu-'):
            allowedDecays.update({'N -> rho+ mu-':'yes'})
        if m > mass('eta1'):
            allowedDecays.update({'N -> eta1 nu_e':'yes'})
            allowedDecays.update({'N -> eta1 nu_mu':'yes'})
            allowedDecays.update({'N -> eta1 nu_tau':'yes'})
        if m > mass('phi'):
            allowedDecays.update({'N -> phi nu_e':'yes'})
            allowedDecays.update({'N -> phi nu_mu':'yes'})
            allowedDecays.update({'N -> phi nu_tau':'yes'})
        if m > mass('e-') + mass('tau-'):
            allowedDecays.update({'N -> e- tau+ nu_tau':'yes'})
            allowedDecays.update({'N -> tau- e+ nu_e':'yes'})
        if m > mass('mu-') + mass('tau-'):
            allowedDecays.update({'N -> mu- tau+ nu_tau':'yes'})
            allowedDecays.update({'N -> tau- mu+ nu_mu':'yes'})
        if m > mass('D_s+') + mass('e-'):
            allowedDecays.update({'N -> D_s+ e-':'yes'})
        if m > mass('D_s+') + mass('mu-'):
            allowedDecays.update({'N -> D_s+ mu-':'yes'})
        if m > mass('D*_s+') + mass('e-'):
            allowedDecays.update({'N -> D*_s+ e-':'yes'})
        if m > mass('D*_s+') + mass('mu-'):
            allowedDecays.update({'N -> D*_s+ mu-':'yes'})
        if m > mass('eta_c'):
            allowedDecays.update({'N -> eta_c nu_e':'yes'})
            allowedDecays.update({'N -> eta_c nu_mu':'yes'})
            allowedDecays.update({'N -> eta_c nu_tau':'yes'})

        for decay in self.decays:
            if decay not in allowedDecays:
                allowedDecays.update({decay:'no'})
        return allowedDecays

    @staticmethod
    def _isVisible(decay):
        return not 'nu' in decay

    def visibleBranchingRatio(self):
        """
        Returns the total HNL branching ratio to visible channels (without neutrinos).
        """
        visibleDecays = filter(self._isVisible, self.decays)
        visibleBR = sum(self.findBranchingRatio(decay) for decay in visibleDecays)
        assert(visibleBR >= 0)
        assert(visibleBR <= 1)
        return visibleBR


class HNL(HNLbranchings):
    """
    HNL physics according to the nuMSM
    """
    def __init__(self, mass, couplings, debug=False):
        """
        Initialize with mass and couplings of the HNL

        Inputs:
        mass (GeV)
        couplings (list of [U2e, U2mu, U2tau])
        """
        HNLbranchings.__init__(self, mass, couplings, debug)
    def computeNLifetime(self,system="SI"):
        """
        Compute the HNL lifetime

        Inputs:
        - system: choose between default (i.e. SI, result in s) or FairShip (result in ns)
        """
        self.NLifetime = c.hGeV / self.NDecayWidth()
        if system == "FairShip": self.NLifetime *= 1.e9
        return self.NLifetime

import matplotlib.pyplot as plt  
import numpy as np

def save_data(masses, branchings, filename):
    data = np.array([masses, branchings])
    np.save(filename, data)

if __name__ == '__main__':
	
    
    #masses = [m for m in np.linspace(0.1,10, 10)]
    #b = [HNL(m,[0,0,1],False).findBranchingRatio('N -> nu nu nu') for m in masses]
    #masses = [1, 1.5, 2, 3, 3.5, 4, 5, 6.5]
    #a = [HNL(m,[1e-6,1e-6,1e-6],False).computeNLifetime()/1e-9 for m in masses]
    #print(a)
    
    # Constants
    vev = 246
    c_SRR = 1e-3 * vev**-2

    # Wilson coefficients simplified
    c_VLR = c_VRR = c_SLR = 0

    # Form factors (as example values)
    f_P = 0.212
    f_P_S = 1

    # Decay width function
    def decay_width(m_P, m_N):
        # Pre-factor
        prefactor = m_P / (32 * np.pi)
        
        # Square root factor
        sqrt_factor = np.sqrt(1 - (4 * m_N**2) / m_P**2)
        
        # First term
        term1 = 2 * np.abs(f_P)**2 * np.abs(c_VRR - c_VLR)**2 * m_N**2
        
        # Second term (with nested parts)
        term2_part1 = np.abs(c_SRR - c_SLR)**2 + np.abs(c_SRR - c_SLR)**2
        term2_part2 = (1 - 2 * m_N**2 / m_P**2)
        term2_part3 = 2 * ((c_SRR - c_SLR) * (c_SRR - c_SLR) + np.conj(c_SRR - c_SLR) * np.conj(c_SRR - c_SLR)) * (m_N**2 / m_P**2)
        term2 = np.abs(f_P_S)**2 * (term2_part1 * term2_part2 + term2_part3)
        
        # Third term (mixed terms)
        term3_part1 = (c_VRR - c_VLR) * (np.conj(c_SRR) - np.conj(c_SLR)) + (c_SRR - c_SLR)
        term3_part2 = (c_VRR - c_VLR) * (c_SRR - c_SLR)
        term3 = f_P * f_P_S * (term3_part1 + term3_part2) * m_N
        
        # Full expression
        gamma = prefactor * sqrt_factor * (term1 + term2 + term3)
        
        return np.real(gamma)  # Taking the real part to avoid small imaginary residues

    # Example plotting over m_N for fixed m_P
    m_N_lifetime = HNL(8,[0,0,1e-7],False).computeNLifetime()
    print(m_N_lifetime*3e8*20)
    #m_N_lifetimes = 2.4e-5

