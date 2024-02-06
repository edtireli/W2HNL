# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

from .channel import (Channel, ChannelList)
from ..util.kinematics import kallen

import numpy as np

class LeptonicChannel(Channel):
    """
    Generic class representing all decay channels involving two leptons in the
    final state.
    """
    def __init__(self, parent, lepton, particle_data):
        super(LeptonicChannel, self).__init__(parent, [lepton], particle_data)

class ChargedLeptonicChannel(LeptonicChannel):
    """
    Decay of charged pseudoscalar meson into a HNL and a charged lepton.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.production.leptonic import ChargedLeptonicChannel
    >>> ch = ChargedLeptonicChannel(411, -13, pdata)
    >>> import hnlbr.util.units as u
    >>> import numpy as np
    >>> hnl_masses = np.array([0.5, 1.0, 1.5, 2.0]) * u.GeV()
    >>> res1 = ch.decay_rate(hnl_masses, [0., 1e-7, 0.])
    >>> target1 = np.array([4.98475158e-29, 1.14026589e-28, 6.21477077e-29, 0.00000000e+00])
    >>> rel_tol = 1e-4
    >>> assert(np.all(np.abs(res1 - target1) <= rel_tol * target1))
    >>> threshold = pdata.mass(411) - pdata.mass(-13)
    >>> hnl_masses_2 = np.array([threshold-1e-6, threshold, threshold+1e-6]) * u.GeV()
    >>> res2 = ch.decay_rate(hnl_masses_2, [0., 1e-7, 0.])
    >>> target2 = np.array([3.91963552e-32, 1.00810454e-36, 0.00000000e+00])
    >>> assert(np.all(np.abs(res2 - target2) <= rel_tol * np.max(target2)))
    >>> ch2 = ChargedLeptonicChannel(211, -11, pdata)
    >>> hnl_masses_3 = np.array([50, 100, 150]) * u.MeV()
    >>> res3 = ch2.decay_rate(hnl_masses_3, [1e-7, 0., 0.])
    >>> target3 = np.array([2.30895324e-31, 2.87846533e-31, 0.00000000e+00])
    >>> assert(np.all(np.abs(res3 - target3) <= rel_tol * target3))
    >>> threshold_2 = pdata.mass(211) - pdata.mass(-11)
    >>> hnl_masses_4 = np.array([threshold_2-u.keV(), threshold_2, threshold_2+u.keV()])
    >>> res4 = ch2.decay_rate(hnl_masses_4, [1e-7, 0., 0.])
    >>> target4 = np.array([7.91744863e-36, 0.00000000e+00, 0.00000000e+00])
    >>> assert(np.all(np.abs(res4 - target4) <= rel_tol * np.max(target4)))
    >>> # Now compare decay rates to PDG ones for light neutrinos.
    >>> res5 = ch2.decay_rate(0., [1., 0., 0.])
    >>> target5 = 3.1735886393504482e-21
    >>> assert(abs(res5 - target5) < rel_tol * target5)
    >>> ch3 = ChargedLeptonicChannel(211, -13, pdata)
    >>> res6 = ch3.decay_rate(0., [0., 1., 0.])
    >>> target6 = 2.4728951645728463e-17
    >>> assert(abs(res6 - target6) < rel_tol * target6)
    """
    def __init__(self, parent, lepton, particle_data):
        super(ChargedLeptonicChannel, self).__init__(parent, lepton, particle_data)

    def decay_rate(self, hnl_mass, couplings=None):
        assert(self.children[1] == self.pdata.hnlid)

        # Prefactor
        fh = self.pdata.pseudoscalar_decay_constants[self.parent]
        mh = self.pdata.mass(self.parent)
        prefactor = self.pdata.GF**2 * fh**2 * mh**3 / (8. * np.pi)

        # Kinematics
        lepton = self.children[0]
        ml = self.pdata.mass(lepton)
        yl = ml / mh
        yN = hnl_mass / mh
        lambda_ = kallen(1., yN**2, yl**2)

        # In principle, either the Källén function or the term with the y's can go negative.
        # In either case, this means that the channel is kinematically closed.
        # Since a closed channel has a zero width, we can handle it by replacing
        # negative values with zero.
        lambda_ = np.clip(lambda_, 0., None)

        # Couplings
        UD = self.pdata.quark_content(self.parent) # E.g. 'ud', 'us', 'cs', ...
        V_UD = self.pdata.ckm[UD]
        U_alpha = self._get_lepton_coupling(lepton, couplings)

        width = ( prefactor * V_UD**2 * U_alpha**2 *
                 (yN**2 + yl**2 - (yN**2 - yl**2)**2) * np.sqrt(lambda_) )
        # Negative width means that the channel is closed. Set its width to zero.
        width = np.clip(width, 0., None)

        return width


class ChargedLeptonicList(ChannelList):
    """
    Class used to generate all the decays of a charged pseudoscalar meson into
    a HNL and a charged lepton.

    `mesons` should be the list of charged pseudoscalar mesons to consider.
    If left unspecified, it defaults to the ones listed in table 7 (left).

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.production.leptonic import ChargedLeptonicList
    >>> chl = ChargedLeptonicList(pdata, mesons=[541, -321])
    >>> from __future__ import print_function
    >>> for ch in chl.list_channels():
    ...     print(ch)
    ...
    321 -> -11 9900015
    321 -> -13 9900015
    321 -> -15 9900015
    541 -> -11 9900015
    541 -> -13 9900015
    541 -> -15 9900015
    -321 -> 11 9900015
    -321 -> 13 9900015
    -321 -> 15 9900015
    -541 -> 11 9900015
    -541 -> 13 9900015
    -541 -> 15 9900015
    """
    def __init__(self, particle_data, mesons=None):
        super(ChargedLeptonicList, self).__init__(particle_data)
        if mesons is None:
            mesons = [211, 321, 411, 431, 521, 541]
        all_mesons = self._add_antiparticles(mesons)
        all_leptons = self._add_antiparticles([11, 13, 15])
        channels = [ChargedLeptonicChannel(meson, lepton, self.pdata)
                        for meson in all_mesons for lepton in all_leptons
                        if self.pdata.charge(meson) == self.pdata.charge(lepton)]
        self.channels = channels

    def list_channels(self):
        """
        Returns the list of leptonic decay channels from a charged pseudoscalar meson.
        """
        return self.channels
