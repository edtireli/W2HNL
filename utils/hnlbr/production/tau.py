# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import scipy.integrate

from .channel import (Channel, ChannelList)
from ..util.kinematics import kallen

class TauDecayChannel(Channel):
    """
    Abstract class representing all tau decay channels producing HNLs.
    """
    def __init__(self, parent, children, particle_data):
        if abs(parent) != 15:
            raise(ArgumentError("Parent must be a tau (anti-)lepton"))
        super(TauDecayChannel, self).__init__(parent, children, particle_data)

    @staticmethod
    def _get_tau_coupling(couplings):
        return couplings[2] if couplings is not None else 1

class TauPseudoscalarDecay(TauDecayChannel):
    """
    τ → N h_P where h_P is a pseudoscalar meson.
    """
    def __init__(self, parent, meson, particle_data):
        super(TauPseudoscalarDecay, self).__init__(parent, [meson], particle_data)
        assert(self.children[1] == self.pdata.hnlid)

    def decay_rate(self, hnl_mass, couplings=None):
        tau = self.parent
        h = self.children[0]

        mtau = self.pdata.mass(tau)
        mh   = self.pdata.mass(h)

        # Decay constant
        fh = self.pdata.pseudoscalar_decay_constants[h]

        # Active-sterile coupling
        U_tau = self._get_tau_coupling(couplings)

        # CKM matrix element
        UD = "".join(self.pdata.quark_content(h))
        V_UD = self.pdata.ckm[UD]

        prefactor = self.pdata.GF**2 * fh**2 * mtau**3 / (16*np.pi)

        yN = hnl_mass / mtau
        yh = mh / mtau

        threshold = mtau - mh

        lambda_ = kallen(1, yN**2, yh**2)
        # Set λ to zero above the closing mass to avoid domain errors
        lambda_ = np.where(hnl_mass < threshold, lambda_, 0.)

        return ( prefactor * V_UD**2 * U_tau**2 * ( (1-yN**2)**2 - yh**2*(1+yN**2) )
                 * np.sqrt(lambda_) )

class TauVectorDecay(TauDecayChannel):
    """
    τ → N h_V where h_V is a vector meson.
    """
    def __init__(self, parent, meson, particle_data):
        super(TauVectorDecay, self).__init__(parent, [meson], particle_data)
        assert(self.children[1] == self.pdata.hnlid)

    def decay_rate(self, hnl_mass, couplings=None):
        tau = self.parent
        h = self.children[0]

        mtau = self.pdata.mass(tau)
        mh   = self.pdata.mass(h)

        # Decay constant
        gh = self.pdata.vector_decay_constants[h]

        # Active-sterile coupling
        U_tau = self._get_tau_coupling(couplings)

        # CKM matrix element
        UD = "".join(self.pdata.quark_content(h))
        V_UD = self.pdata.ckm[UD]

        prefactor = self.pdata.GF**2 * gh**2 * mtau**3 / (16*np.pi * mh**2)

        yN = hnl_mass / mtau
        yh = mh / mtau

        threshold = mtau - mh

        lambda_ = kallen(1, yN**2, yh**2)
        # Set λ to zero above the closing mass to avoid domain errors
        lambda_ = np.where(hnl_mass < threshold, lambda_, 0.)

        return ( prefactor * V_UD**2 * U_tau**2 * ( (1-yN**2)**2 + yh**2*(1 + yN**2 - 2*yh**2) )
                 * np.sqrt(lambda_) )

class TauNuLDecay(TauDecayChannel):
    """
    τ → N l ν̄_l, where l ∈ {e,μ}
    """
    def __init__(self, parent, charged_lepton, particle_data):
        neutrino = -np.sign(charged_lepton) * (abs(charged_lepton)+1)
        super(TauNuLDecay, self).__init__(parent, [charged_lepton, neutrino], particle_data)
        assert(charged_lepton * neutrino < 0)
        assert(abs(neutrino) == abs(charged_lepton)+1)
        assert(abs(charged_lepton) in [11, 13])

    def decay_rate(self, hnl_mass, couplings=None):
        assert(abs(self.children[0]) in [11, 13])

        mtau = self.pdata.mass(self.parent)
        ml   = self.pdata.mass(self.children[0])

        U_tau = self._get_tau_coupling(couplings)

        prefactor = self.pdata.GF**2 * mtau**5 / (96*np.pi**3)

        def integrand(x, yN, yl):
            return (x-yl**2)**2 * np.sqrt(kallen(1,x,yN**2)) * (
                (x+2*yl**2)*(1-yN**2)**2 + x*(x-yl**2)*(1+yN**2-yl**2) - x*yl**4 - 2*x**3
            ) / x**3

        def lower_limit(yN, yl):
            return yl**2

        def upper_limit(yN, yl):
            return (1-yN)**2

        max_hnl_mass = mtau - ml

        def decay_rate(mN):
            if mN < max_hnl_mass:
                yN = mN / mtau
                yl = ml / mtau
                integral, err = scipy.integrate.quad(
                    lambda x: integrand(x, yN, yl),
                    lower_limit(yN, yl),
                    upper_limit(yN, yl)
                )
                return prefactor * U_tau**2 * integral
            else:
                return 0.

        return np.vectorize(decay_rate)(hnl_mass)

class TauNuTauDecay(TauDecayChannel):
    """
    τ → ν̄_τ l N, where l ∈ {e,μ}
    """
    def __init__(self, parent, charged_lepton, particle_data):
        neutrino = +np.sign(parent) * (abs(parent)+1)
        super(TauNuTauDecay, self).__init__(parent, [charged_lepton, neutrino], particle_data)
        assert(parent * neutrino > 0)
        assert(abs(neutrino) == 16)
        assert(abs(charged_lepton) in [11, 13])

    def decay_rate(self, hnl_mass, couplings=None):
        charged_lepton = self.children[0]

        mtau = self.pdata.mass(self.parent)
        ml   = self.pdata.mass(charged_lepton)

        # Here the coupling is between the non-τ lepton and the HNL
        U_alpha = self._get_lepton_coupling(charged_lepton, couplings)

        prefactor = self.pdata.GF**2 * mtau**5 / (96*np.pi**3)

        def integrand(x, yN, yl):
            return (1-x)**2 * np.sqrt(kallen(x,yN**2,yl**2)) * (
                2*x**3 + x - x*(1-x)*(1-yN**2-yl**2) - (2+x)*(yN**2-yl**2)**2
            ) / x**3

        def lower_limit(yN, yl):
            return (yl+yN)**2

        def upper_limit(yN, yl):
            return 1.

        max_hnl_mass = mtau - ml

        def decay_rate(mN):
            if mN < max_hnl_mass:
                yN = mN / mtau
                yl = ml / mtau
                integral, err = scipy.integrate.quad(
                    lambda x: integrand(x, yN, yl),
                    lower_limit(yN, yl),
                    upper_limit(yN, yl)
                )
                return prefactor * U_alpha**2 * integral
            else:
                return 0.

        return np.vectorize(decay_rate)(hnl_mass)
