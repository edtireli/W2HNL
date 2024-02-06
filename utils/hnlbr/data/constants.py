# -*- coding: utf-8 -*-

# Source:
# Review of Particle Physics (2016)
# 1. Physical Constants

from __future__ import absolute_import

import math

from ..util import units as u

def GF():
    """
    Fermi constant in natural units
    G_F = 1.166 378 7(6)×10⁻⁵ GeV⁻²

    >>> from hnlbr.data.constants import GF
    >>> GF()
    1.166378e-05
    """
    return 1.166378e-5 / u.GeV()**2

def s2thetaW():
    """
    sin(θ_W)² = 0.23129(5) (MSbar scheme)

    >>> from hnlbr.data.constants import s2thetaW
    >>> s2thetaW()
    0.23129
    """
    return 0.23129

def thetaW():
    """
    Weinberg's angle θ_W

    >>> from hnlbr.data.constants import thetaW
    >>> thetaW()
    0.5017107831701941
    """
    s2 = s2thetaW()
    return math.asin(math.sqrt(s2))
