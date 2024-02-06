# -*- coding: utf-8 -*-

from __future__ import division

def GeV():
    return 1.

def MeV():
    return 1e-3 * GeV()

def keV():
    return 1e-6 * GeV()

def eV():
    return 1e-9 * GeV()

def TeV():
    return 1e3 * GeV()

def PeV():
    return 1e6 * GeV()

def s_per_m():
    return 299792458

def MeV_s():
    return 1. / 6.582119514e-22

def s():
    """
    Second in GeV⁻¹

    >>> from hnlbr.util.units import *
    >>> x = 6.582119514e-22 * MeV() * s()
    >>> assert(abs(x - 1.) < 1e-14)
    """
    return MeV_s() / MeV()

def m():
    """
    Meter in GeV⁻¹

    >>> from hnlbr.util.units import *
    >>> x = 299792458. * m() / s()
    >>> assert(abs(x - 1.) < 1e-14)
    """
    return s() / s_per_m()
