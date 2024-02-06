# -*- coding: utf-8 -*-

# Sources:
# ========
#
# Data:
# -----
# Revisiting SHiP prospects in searches for Heavy Neutral Leptons
# TODO: add author list
# See references in this paper for the primary sources.
#
# PDG IDs:
# --------
# Review of Particle Physics (2016)
# 44. Monte Carlo particle numbering scheme
# Revised October 2017 by L. Garren (Fermilab), F. Krauss (Durham U.),
# C.-J. Lin (LBNL), S. Navas (U. Granada), P. Richardson (Durham U.),
# and T. Sjöstrand (Lund U.).

from __future__ import absolute_import

import math

from ..util import units as u

class DecayConstants(object):
    """
    Abstract class representing a list of meson decay constants.
    """
    def __init__(self):
        pass

    def get(self, pdgid):
        """
        Returns the decay constant corresponding to the particle's PDG ID

        >>> from hnlbr.data.decay_constants import PseudoscalarDC
        >>> dc = PseudoscalarDC()
        >>> dc.get(441)
        0.335
        """
        if isinstance(pdgid, int):
            return self._dict[abs(pdgid)]
        else:
            raise ValueError("PDG ID must be an integer")

    def __getitem__(self, pdgid):
        """
        Returns the decay constant corresponding to the particle's PDG ID

        >>> from hnlbr.data.decay_constants import (PseudoscalarDC, VectorDecayConstants)
        >>> dc = PseudoscalarDC()
        >>> dc[111]
        0.13019999999999998
        >>> dc[441]
        0.335
        >>> dc[221]
        -0.08170000000000001
        >>> dcv = VectorDecayConstants()
        >>> dcv.get(213)
        0.162
        >>> dcv.get(-413)
        0.535
        >>> dcv.get(433)
        0.65
        """
        return self.get(pdgid)

class PseudoscalarDC(DecayConstants):
    """
    Class containing the decay constants for pseudoscalar mesons (either charged or neutral)
    """
    def __init__(self):
        super(PseudoscalarDC, self).__init__()
        self._dict = {
            211: 130.2 * u.MeV(),   # π⁺
            321: 155.6 * u.MeV(),   # K⁺
            411: 212. * u.MeV(),    # D⁺
            431: 249. * u.MeV(),    # Ds⁺
            521: 187. * u.MeV(),    # B⁺
            541: 434. * u.MeV(),    # Bc⁺
            111: 130.2 * u.MeV(),   # π⁰
            221: (-81.7 * u.MeV()), # η
            331: 94.7 * u.MeV(),    # η' (958)
            441: 335. * u.MeV()     # ηc (1S)
        }

class VectorDecayConstants(DecayConstants):
    """
    Class containing the decay constants for charged vector mesons.
    """
    def __init__(self):
        super(VectorDecayConstants, self).__init__()
        self._dict = {
            213: 0.162 * u.GeV()**2, # ρ⁺
            413: 0.535 * u.GeV()**2, # D⁺*
            433: 0.650 * u.GeV()**2  # Ds⁺*
        }
