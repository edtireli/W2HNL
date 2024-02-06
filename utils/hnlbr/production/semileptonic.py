# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from future.utils import viewitems

from collections import Counter
import numpy as np
import scipy.integrate

from .channel import (Channel, ChannelList)
from ..util.kinematics import kallen

class SemileptonicChannel(Channel):
    """
    Abstract class representing all semi-leptonic decay channels of a meson into
    another meson, a charged lepton and a HNL.
    """
    def __init__(self, parent, meson, lepton, particle_data):
        super(SemileptonicChannel, self).__init__(parent, [meson, lepton], particle_data)

    def _get_clebsch_gordan(self, h):
        # FIXME: come up with a more generic solution.
        return (1. / np.sqrt(2.)) if h in [111, 113] else 1.

    def _quark_difference(self, qq1, qq2):
        # We need to figure out which quark decays
        # Let's use a multiset for that
        qh1 = Counter(qq1)
        qh2 = Counter(qq2)
        # This should contain the parent quark with count +1, and the children
        # quark with count -1.
        diff = qh2.copy()
        diff.subtract(qh1)
        quarks = []
        for (k, v) in viewitems(diff):
            if v != 0:
                quarks.append(k)
        return quarks

    def _get_ckm_element(self, h1, h2):
        if (h2 // 10) in [11, 22, 33]: # Special case for light unflavored mesons: π⁰, η, …
            diff1 = self._quark_difference(self.pdata.quark_content(h1), ["u", "u"])
            diff2 = self._quark_difference(self.pdata.quark_content(h1), ["d", "d"])
            diff3 = self._quark_difference(self.pdata.quark_content(h1), ["s", "s"])
            if len(diff1) == 2:
                diff = diff1
            elif len(diff2) == 2:
                diff = diff2
            elif len(diff3) == 2:
                diff = diff3
            else:
                raise RuntimeError("Failed to compute quark difference between " +
                                   str(h1) + "and pi0")
        else:
            diff = self._quark_difference(self.pdata.quark_content(h1), self.pdata.quark_content(h2))

        if len(diff) != 2:
            raise ValueError("Failed to compute CKM element for " + str(h1) + " -> " + str(h2))

        V_UD = self.pdata.ckm["".join(diff)]
        return V_UD

    # Utility functions
    @staticmethod
    def Lambda(xi, yh, yN, yl):
        return np.sqrt(kallen(1, yh**2, xi) * kallen(xi, yN**2, yl**2))

    @staticmethod
    def F(xi, yh):
        return (1-xi)**2 - 2*yh**2*(1+xi) + yh**4

    @staticmethod
    def GM(xi, yN, yl):
        return xi*(yN**2+yl**2) - (yN**2-yl**2)**2

    @staticmethod
    def GP(xi, yN, yl):
        return xi*(yN**2+yl**2) + (yN**2-yl**2)**2

    # Integration limits
    @staticmethod
    def _lower_limit(yh, yN, yl):
        return (yl+yN)**2

    @staticmethod
    def _upper_limit(yh, yN, yl):
        return (1.-yh)**2

    @staticmethod
    def _integrate(func, yh, yN, yl):
        val, err = scipy.integrate.quad(
            lambda x: func(x, yh, yN, yl),
            SemileptonicChannel._lower_limit(yh, yN, yl),
            SemileptonicChannel._upper_limit(yh, yN, yl)
        )
        return val


class SemileptonicPSChannel(SemileptonicChannel):
    """
    Decay of a meson into a HNL, a pseudoscalar meson and a charged lepton.
    """
    def __init__(self, parent, meson, lepton, particle_data):
        super(SemileptonicPSChannel, self).__init__(parent, meson, lepton, particle_data)

    def decay_rate(self, hnl_mass, couplings=None):
        assert(self.children[2] == self.pdata.hnlid)

        h1 = self.parent # Incoming meson
        h2 = self.children[0] # Outgoing meson
        lepton  = self.children[1] # Outgoing lepton

        mh1 = self.pdata.mass(h1)
        mh2 = self.pdata.mass(h2)
        ml  = self.pdata.mass(lepton)

        # Prefactor
        prefactor = self.pdata.GF**2 * mh1**5 / (64. * np.pi**3)

        # Clebsch-Gordan coefficient (1/√2 for π⁰ and ρ⁰, 1 otherwise)
        cg = self._get_clebsch_gordan(h2)

        # Couplings
        # ---------

        # PMNS coupling is trivial
        U_alpha = self._get_lepton_coupling(lepton, couplings)

        # CKM coupling
        V_UD = self._get_ckm_element(h1, h2)

        # Integrals
        # ---------

        # Form factors
        # Here, form factors are actually real, so we can omit the modulus
        # TODO: check the statement below
        # We assume that excited states have the same form factors as ground states
        fP, f0 = self.pdata.ffdb[np.sign(h1) * (abs(h1) % 10000), np.sign(h2) * (abs(h2) % 10000)]

        # Integrands
        def int_1(xi, yh, yN, yl):
            return fP(xi*mh1**2)**2 * self.Lambda(xi, yh, yN, yl)**3 / (3. * xi**3)

        def int_2(xi, yh, yN, yl):
            return ( fP(xi*mh1**2)**2 * self.Lambda(xi, yh, yN, yl) *
                     self.GM(xi, yN, yl) * kallen(1., yh**2, xi) / (2. * xi**3) )

        def int_3(xi, yh, yN, yl):
            return ( f0(xi*mh1**2)**2 * self.Lambda(xi, yh, yN, yl) *
                     self.GM(xi, yN, yl) * (1.-yh**2)**2 / (2. * xi**3) )

        # See https://stackoverflow.com/a/34021333 for why the double lambda is needed
        IP1, IP2, IP3 = [(lambda _func:(lambda yh, yN, yl: self._integrate(_func, yh, yN, yl)))(func)
                         for func in [int_1, int_2, int_3]]

        # Decay rate
        # ----------

        max_hnl_mass = mh1 - mh2 - ml

        def decay_rate(mN):
            if mN < max_hnl_mass:
                return prefactor * cg**2 * V_UD**2 * U_alpha**2 * (
                    IP1(mh2/mh1, mN/mh1, ml/mh1) + IP2(mh2/mh1, mN/mh1, ml/mh1) +
                    IP3(mh2/mh1, mN/mh1, ml/mh1) )
            else:
                return 0.

        return np.vectorize(decay_rate)(hnl_mass)


class SemileptonicPSChannelList(ChannelList):
    def __init__(self, particle_data):
        super(SemileptonicPSChannelList, self).__init__(particle_data)
        mesons = [
            # Kaon decays
            ["K0", "pi-"],
            ["K+", "pi0"],
            # D meson decays
            ["D0", "K-"],
            ["D+", "Kbar0"],
            # ["D_s+", "K0"],
            ["D_s+", "eta"],
            # B meson decays
            ["B0", "D-"],
            ["B+", "Dbar0"],
            ["B_s0", "D_s-"],
            ["B_s0", "K-"],
            ["B0", "pi-"],
            ["B+", "pi0"]
        ]
        processes = [[meson[0], meson[1], lepton]
                     for lepton in ["e+", "mu+", "tau+"]
                     for meson in mesons]
        process_ids = self._add_conjugated_channels(
            [[self.pdata.pdg_id(p) for p in proc] for proc in processes])
        self.channels = [SemileptonicPSChannel(proc[0], proc[1], proc[2], self.pdata)
                    for proc in process_ids]

    def list_channels(self):
        """
        Returns the list of semileptonic decay channels of a meson to a
        pseudoscalar meson.
        """
        return self.channels


class SemileptonicVectorChannel(SemileptonicChannel):
    """
    Decay of a meson into a HNL, a vector meson and a charged lepton.
    """
    def __init__(self, parent, meson, lepton, particle_data):
        super(SemileptonicVectorChannel, self).__init__(parent, meson, lepton, particle_data)

    def decay_rate(self, hnl_mass, couplings=None):
        assert(self.children[2] == self.pdata.hnlid)

        h  = self.parent # Incoming meson
        h1 = self.children[0] # Outgoing meson
        lepton  = self.children[1] # Outgoing lepton

        mh  = self.pdata.mass(h)
        mh1 = self.pdata.mass(h1)
        ml  = self.pdata.mass(lepton)

        prefactor = ( self.pdata.GF**2 * mh**7 ) / ( 64*np.pi**3 * mh1**2 )

        cg = self._get_clebsch_gordan(h1)
        U_alpha = self._get_lepton_coupling(lepton, couplings)
        V_UD = self._get_ckm_element(h, h1)

        # Dimensionful vector form factors
        f, g, aP, aM = self.pdata.ffdb_vec.make_dimensionful_ff(h, h1)

        # Integrals
        def int_g2(x, yh, yN, yl):
            return ( (mh**2 * yh**2 / 3) * g(x*mh**2)**2 * self.Lambda(x, yh, yN, yl)
                     * self.F(x, yh) * (2*x**2 - self.GP(x, yN, yl)) / x**2 )

        def int_f2(x, yh, yN, yl):
            return 1/(24*mh**2) * f(x*mh**2)**2 * self.Lambda(x, yh, yN, yl) * (
                3*self.F(x, yh)*(x**2-(yl**2-yN**2)**2) - self.Lambda(x, yh, yN, yl)**2
                + 12*yh**2*x*(2*x**2 - self.GP(x, yN, yl)) ) / x**3

        def int_aP2(x, yh, yN, yl):
            return ( mh**2/24 * aP(x*mh**2)**2 * self.Lambda(x, yh, yN, yl) * self.F(x, yh) * (
                self.F(x, yh) * (2*x**2 - self.GP(x, yN, yl)) + 3*self.GM(x, yN, yl) * (1-yh**2)**2
            ) / x**3 )

        def int_aM2(x, yh, yN, yl):
            return ( mh**2/8 * aM(x*mh**2)**2 * self.Lambda(x, yh, yN, yl) * self.F(x, yh)
                     * self.GM(x, yN, yl) / x )

        def int_f_aP(x, yh, yN, yl):
            return 1/12 * f(x*mh**2) * aP(x*mh**2) * self.Lambda(x, yh, yN, yl) * (
                3*x*self.F(x, yh)*self.GM(x, yN, yl) + (1-x-yh**2) * (
                    3*self.F(x, yh)*(x**2-(yl**2-yN**2)**2) - self.Lambda(x, yh, yN, yl)**2 )
            ) / x**3

        def int_f_aM(x, yh, yN, yl):
            return ( 1/4 * f(x*mh**2) * aM(x*mh**2) * self.Lambda(x, yh, yN, yl)
                     * self.F(x, yh) * self.GM(x, yN, yl) / x**2 )

        def int_aP_aM(x, yh, yN, yl):
            return ( mh**2/4 * aP(x*mh**2) * aM(x*mh**2) * self.Lambda(x, yh, yN, yl)
                     * self.F(x, yh) * self.GM(x, yN, yl) * (1-yh**2) / x**2 )

        integrands = [int_g2, int_f2, int_aP2, int_aM2, int_f_aP, int_f_aM, int_aP_aM]

        # See https://stackoverflow.com/a/34021333 for why the double lambda is needed
        IVs = [(lambda _func:(lambda yh, yN, yl: self._integrate(_func, yh, yN, yl)))(func)
                         for func in integrands]
        # IV_g2, IV_f2, IV_aP2, IV_aM2, IV_f_aP, IV_f_aM, IV_aP_aM = IVs

        # Decay rate

        max_hnl_mass = mh - mh1 - ml

        def decay_rate(mN):
            if mN < max_hnl_mass:
                return ( prefactor * cg**2 * V_UD**2 * U_alpha**2 *
                         sum(IV(mh1/mh, mN/mh, ml/mh) for IV in IVs) )
            else:
                return 0.

        return np.vectorize(decay_rate)(hnl_mass)

class SemileptonicVectorChannelList(ChannelList):
    def __init__(self, particle_data):
        super(SemileptonicVectorChannelList, self).__init__(particle_data)
        mesons = [
            # D → K*
            ["D0", "K*-"],
            ["D+", "K*bar0"],
            # B → D*
            ["B0", "D*-"],
            ["B+", "D*bar0"],
            # B → ρ
            ["B0", "rho-"],
            ["B+", "rho0"],
            # Bs → Ds*
            ["B_s0", "D*_s-"],
            # Bs → K*
            ["B_s0", "K*-"]
        ]
        processes = [[meson[0], meson[1], lepton]
                     for lepton in ["e+", "mu+", "tau+"]
                     for meson in mesons]
        process_ids = self._add_conjugated_channels(
            [[self.pdata.pdg_id(p) for p in proc] for proc in processes])
        self.channels = [SemileptonicVectorChannel(proc[0], proc[1], proc[2], self.pdata)
                    for proc in process_ids]

    def list_channels(self):
        """
        Returns the list of semileptonic decay channels of a meson to a
        pseudoscalar meson.
        """
        return self.channels
