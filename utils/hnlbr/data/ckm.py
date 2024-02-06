# -*- coding: utf-8 -*-

# Absolute values and errors of CKM matrix elements
def ud():
    """|V_ud|"""
    return 0.97417

def err_ud():
    """δ|V_ud|"""
    return 0.00021

def us():
    """|V_us|"""
    return 0.2248

def err_us():
    """δ|V_us|"""
    return 0.0006

def cd():
    """|V_cd|"""
    return 0.220

def err_cd():
    """δ|V_cd|"""
    return 0.005

def cs():
    """|V_cs|"""
    return 0.995

def err_cs():
    """δ|V_cs|"""
    return 0.016

def cb():
    """|V_cb|"""
    return 40.5e-3

def err_cb():
    """δ|V_cb|"""
    return 1.5e-3

def ub():
    """|V_ub|"""
    return 4.09e-3

def err_ub():
    """δ|V_ub|"""
    return 0.39e-3

def td():
    """|V_td|"""
    return 8.2e-3

def err_td():
    """δ|V_td|"""
    return 0.6e-3

def ts():
    """|V_ts|"""
    return 40.0e-3

def err_ts():
    """δ|V_ts|"""
    return 2.7e-3

def tb():
    """|V_tb|"""
    return 1.0 # Best-fit is actually 1.009

def err_tb():
    """δ|V_tb|"""
    return 0.031

class CKMabs(object):
    """
    Helper class to query the absolute values of the CKM matrix elements.

    Source:
    Review of Particle Physics (2016).
    12. CKM Quark-Mixing Matrix
    Revised January 2016 by A. Ceccucci (CERN), Z. Ligeti (LBNL), and Y. Sakai (KEK).
    """
    def __init__(self):
        self._dict = {
            "ud": ud(),
            "us": us(),
            "ub": ub(),
            "cd": cd(),
            "cs": cs(),
            "cb": cb(),
            "td": td(),
            "ts": ts(),
            "tb": tb()
        }
        self._err_dict = {
            "ud": err_ud(),
            "us": err_us(),
            "ub": err_ub(),
            "cd": err_cd(),
            "cs": err_cs(),
            "cb": err_cb(),
            "td": err_td(),
            "ts": err_ts(),
            "tb": err_tb()
        }

    def get(self, U, D):
        """
        Helper method to query absolute value of matrix element using flavor

        >>> from hnlbr.data.ckm import CKMabs
        >>> ckm = CKMabs()
        >>> ckm.get('u', 's')
        0.2248
        """
        return self._dict[U+D]

    def get_err(self, U, D):
        """
        Helper method to query error on matrix element using flavor

        >>> from hnlbr.data.ckm import CKMabs
        >>> ckm = CKMabs()
        >>> ckm.get_err('u', 's')
        0.0006
        """
        return self._err_dict[U+D]

    def __getitem__(self, idx):
        """
        Query absolute value of matrix element using flavor.

        >>> from hnlbr.data.ckm import CKMabs
        >>> ckm = CKMabs()
        >>> ckm['u', 'd']
        0.97417
        >>> ckm['s', 'u']
        0.2248

        Can be called either as ckm[U, D] or as ckm[D, U].
        U must be one of 'u', 'c', 't'.
        D must be one of 'd', 's', 'b'.
        """
        if idx[0] in ['u', 'c', 't']:
            U, D = idx
        else:
            D, U = idx
        return self.get(U, D)
