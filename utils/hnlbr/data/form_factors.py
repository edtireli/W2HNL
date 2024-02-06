# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from future.utils import viewitems

import numpy as np
import copy

from ..util import units as u

def z(q2, m1, m2):
    tp = (m1 + m2)**2
    t0 = (m1 + m2) * (np.sqrt(m1) - np.sqrt(m2))**2
    return (np.sqrt(tp - q2) - np.sqrt(tp - t0)) / (np.sqrt(tp - q2) + np.sqrt(tp - t0))

class FormFactor(object):
    """
    Abstract class representing a meson form factor.
    """
    def __init__(self):
        pass

    def __call__(self, q2):
        """
        Evaluates the form factor at the given q².

        It is expected from subclasses to implement this method in a vectorized way.
        """
        raise RuntimeError("Call operator must be defined by concrete class")

    @classmethod
    def form_factors(cls, pdata):
        """
        Abstract method returning the list of channels for which the expression
        of the form factor is valid, along with the corresponding form factors.

        Returns:
        A dictionary with keys corresponding to the tuples (h, h') of mesons
        involved in the process, and values the tuples of form factors (f₊, f₀).
        """
        raise RuntimeError("Static method must be defined by subclasses")

class KaonFormFactor(FormFactor):
    """
    Form factor for the K → π transition.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.data.form_factors import KaonFormFactor
    >>> ff0 = KaonFormFactor(0.970, 0.0117, pdata.mass(211))
    >>> import numpy as np
    >>> q2 = np.linspace(0., 1., 6)
    >>> res = ff0(q2)
    >>> target = np.array([0.97, 1.08652079, 1.20304159, 1.31956238, 1.43608318, 1.55260397])
    >>> rel_tol = 1e-2
    >>> assert(np.all(np.abs(res - target) < rel_tol * target))
    """
    def __init__(self, f0, lambda_, m_pi):
        super(KaonFormFactor, self).__init__()
        self.f0 = f0
        self.l = lambda_
        self.m_pi = m_pi

    def __call__(self, q2):
        return self.f0 * (1. + self.l * (q2 / self.m_pi**2))

    @classmethod
    def form_factors(cls, pdata):
        m_pi = pdata.mass(211)
        return {
            # Data from table 10
            # K⁰ → π⁻
            (311, -211): (cls(0.970, 0.0267, m_pi), cls(0.970, 0.0117, m_pi)),
            # K⁺ → π⁰
            (321, 111): (cls(0.970, 0.0277, m_pi), cls(0.970, 0.0183, m_pi))
        }


class DMesonFormFactor(FormFactor):
    """
    Form factor for the D → K and D → π transitions.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.data.form_factors import DMesonFormFactor
    >>> ff = DMesonFormFactor(0.7647, 0.066, 0.224, pdata.mass(421), pdata.mass(321))
    >>> import numpy as np
    >>> q2 = np.linspace(0., (pdata.mass(421)-pdata.mass(321))**2, 6)
    >>> res = ff(q2)
    >>> target = np.array([0.7647, 0.83634761, 0.92259383, 1.02838631, 1.16118918, 1.33281503])
    >>> rel_tol = 1e-3
    >>> assert(np.all(np.abs(res - target) < rel_tol * target))
    """
    def __init__(self, f0, c, P, mh, mh_prime):
        super(DMesonFormFactor, self).__init__()
        self.f0 = f0
        self.c = c
        self.P = P
        self.m1 = mh
        self.m2 = mh_prime

    def __call__(self, q2):
        z0 = z(0, self.m1, self.m2)
        zq2 = z(q2, self.m1, self.m2)
        return (self.f0 - self.c * (zq2 - z0) * (1 + (zq2 + z0) / 2.)) / (1 - self.P * q2)

    @classmethod
    def form_factors(cls, pdata):
        # Data from table 11
        def fDK(p1, p2):
            return (cls(0.7647, 0.066, 0.224, pdata.mass(p1), pdata.mass(p2)),
                    cls(0.7647, 2.084, 0.   , pdata.mass(p1), pdata.mass(p2)))
        def fDpi(p1, p2):
            return (cls(0.6117, 1.985, 0.1314, pdata.mass(p1), pdata.mass(p2)),
                    cls(0.6117, 1.188, 0.0342, pdata.mass(p1), pdata.mass(p2)))
        return {
            # D⁺ → K̄⁰
            (411, -311): fDK(411, 311),
            # D⁰ → K⁻
            (421, -321): fDK(421, 321),
            # D⁺ → π⁰
            (411, 111): fDpi(411, 111),
            # D⁰ → π⁻
            (421, -211): fDpi(421, 211)
        }


class DsMesonFormFactor(FormFactor):
    """
    Form factor for the Ds → η transition.

    >>> from hnlbr.data.form_factors import DsMesonFormFactor
    >>> ff = DsMesonFormFactor(0.495, 0.198, 2.112)
    >>> import numpy as np
    >>> q2 = np.linspace(0., 1., 6)
    >>> res = ff(q2)
    >>> target = np.array([0.495, 0.5228785, 0.55359133, 0.58758153, 0.62538906, 0.66767883])
    >>> rel_tol = 1e-3
    >>> assert(np.all(np.abs(res - target) < rel_tol * target))
    """
    def __init__(self, f0, alpha, m_Ds_star):
        super(DsMesonFormFactor, self).__init__()
        self.f0 = f0
        self.alpha = alpha
        self.mh =  m_Ds_star

    def __call__(self, q2):
        return self.f0 / ((1 - q2 / self.mh**2) * (1 - self.alpha * (q2 / self.mh**2)))

    @classmethod
    def form_factors(cls, pdata):
        # Data from section C.2.2
        m_Ds_star = 2.112 * u.GeV()
        return {
            # Ds → η
            (431, 221): (cls(0.495, 0.198, m_Ds_star), cls(0.495, 0., m_Ds_star))
        }

class BMesonFormFactor(FormFactor):
    """
    Abstract class representing a generic B meson form factor.
    """
    def __init__(self):
        super(BMesonFormFactor, self).__init__()

    # Data from tables 12 and 13
    @staticmethod
    def _fBD(p1, p2, pdata):
        mB = pdata.mass(p1)
        mD = pdata.mass(p2)
        return (BMesonGenericFF(float("inf"), 0.890, -8.47, 39, mB, mD),
                # There might be a sign error here (-12 → +12),
                # but it makes no difference in practice so I leave it as-is.
                BMesonGenericFF(float("inf"), 0.774, -3.64, -12, mB, mD))
    @staticmethod
    def _fBsK(p1, p2, pdata):
        mBs = pdata.mass(p1)
        mK = pdata.mass(p2)
        return (BMesonGenericFF(5.325, 0.363, -0.78, 1.9, mBs, mK),
                BMesonGenericFF(5.65, 0.210, -0.21, -1.4, mBs, mK))

    @staticmethod
    def _fBpi(p1, p2, pdata):
        mB = pdata.mass(p1)
        mpi = pdata.mass(p2)
        return (BMesonGenericFF(5.325, 0.421, -0.35, -0.41, mB, mpi),
                BMesonToPiFF0(0.507, -1.77, 1.3, 4, mB, mpi)) # N_z = 4 fit
                #BMesonToPiFF0(0.515, -1.84, -0.14, mB, mpi)) # N_z = 3 fit (gives negative values)

    @classmethod
    def old_form_factors(cls, pdata):
        """
        Uses the data from the FLAG preprint (1607.00299)
        """
        return {
            (521, -421): cls._fBD(521, 421, pdata),  # B⁺ → D̄⁰
            (511, -411): cls._fBD(511, 411, pdata),  # B⁰ → D⁻
            (531, -431): cls._fBD(531, 431, pdata),  # Bₛ⁰ → Dₛ⁻
            (531, -321): cls._fBsK(531, 321, pdata), # Bₛ⁰ → K⁻
            (521,  111): cls._fBpi(521, 111, pdata), # B⁺ → π⁰
            (511, -211): cls._fBpi(511, 211, pdata)  # B⁰ → π⁻
        }

    @staticmethod
    def _aP_Bpi():
        return [0.404, -0.68, -0.86]

    @staticmethod
    def _a0_Bpi():
        return [0.490, -1.61]

    @staticmethod
    def _aP_BK():
        return [0.360, -0.828, 1.11]

    @staticmethod
    def _a0_BK():
        return [0.233, 0.197]

    @staticmethod
    def _aP_BD():
        return [0.909, -7.11, 66]

    @staticmethod
    def _a0_BD():
        return [0.794, -2.45]

    @staticmethod
    def _mBStar():
        return 5.325 * u.GeV()

    @classmethod
    def new_form_factors(cls, pdata):
        """
        Uses the data from the journal version of the FLAG paper (10.1140/epjc/s10052-016-4509-7)
        """
        fit_coefficients = [
            (511, -211, cls._aP_Bpi(), cls._a0_Bpi()), # B⁰ → π⁻
            (521,  111, cls._aP_Bpi(), cls._a0_Bpi()), # B⁺ → π⁰
            (531, -321, cls._aP_BK() , cls._a0_BK() ), # Bₛ⁰ → K⁻
            (511, -411, cls._aP_BD() , cls._a0_BD() ), # B⁰ → D⁻
            (521, -421, cls._aP_BD() , cls._a0_BD() ), # B⁺ → D̄⁰
            (531, -431, cls._aP_BD() , cls._a0_BD() )  # Bₛ⁰ → Dₛ⁻
        ]
        return { (h, h_prime): BMesonBCLFormFactor.make_pair(
            aP, a0, pdata.mass(h), pdata.mass(h_prime), cls._mBStar(), float("inf"))
                 for (h, h_prime, aP, a0) in fit_coefficients }

    @classmethod
    def best_form_factors(cls, pdata):
        """
        Uses the form factors which give the best agreement between the
        computed branching ratios and the ones tabulated in the PDG for M=0.
        """
        return {
            (521, -421): cls._fBD(521, 421, pdata),  # B⁺ → D̄⁰
            (511, -411): cls._fBD(511, 411, pdata),  # B⁰ → D⁻
            (531, -431): cls._fBD(531, 431, pdata),  # Bₛ⁰ → Dₛ⁻
            (531, -321): cls._fBsK(531, 321, pdata), # Bₛ⁰ → K⁻
            (521,  111): BMesonBCLFormFactor.make_pair(
                cls._aP_Bpi(), cls._a0_Bpi(), pdata.mass(521), pdata.mass(111),
                cls._mBStar(), float("inf"), constrained=False), # B⁺ → π⁰
            (511, -211): BMesonBCLFormFactor.make_pair(
                cls._aP_Bpi(), cls._a0_Bpi(), pdata.mass(511), pdata.mass(211),
                cls._mBStar(), float("inf"), constrained=False)  # B⁰ → π⁻
        }

class BMesonBCLFormFactor(BMesonFormFactor):
    """
    Generic B meson form factor, expressed using the BCL parametrization.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.data.form_factors import BMesonBCLFormFactor
    >>> mB = pdata.mass(511)
    >>> mD = pdata.mass(411)
    >>> mBStar = 5.325 * u.GeV()
    >>> ff = BMesonBCLFormFactor([0.909, -7.11, 66], mB, mD, mBStar)
    >>> import numpy as np
    >>> q2 = np.linspace(0., (mB - mD)**2, 6)
    >>> res = ff(q2)
    >>> target = np.array([0.74833405, 0.86107096, 1.0213944, 1.24908075, 1.57445903, 2.04599588])
    >>> rel_tol = 1e-2
    >>> assert(np.all(np.abs(res - target) < rel_tol * target))
    """
    def __init__(self, a, mh, mh_prime, M_pole=float("+inf")):
        super(BMesonFormFactor, self).__init__()
        self.a = copy.copy(a)
        self.mh = mh
        self.mh_prime = mh_prime
        self.M_pole = M_pole

    def __call__(self, q2):
        zq2 = z(q2, self.mh, self.mh_prime)
        return sum( self.a[n] * zq2**n for n in range(len(self.a)) ) / (1. - q2/self.M_pole**2)

    @classmethod
    def make_pair(cls, aP, a0, mh, mh_prime, mpoleP, mpole0, constrained=True):
        """
        Instantiates form factor functors f0 and f+ from the 5-parameter joint fit.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> from hnlbr.data.form_factors import BMesonBCLFormFactor
        >>> mB = pdata.mass(511)
        >>> mD = pdata.mass(411)
        >>> from hnlbr.util import units as u
        >>> mBStar = 5.325 * u.GeV()
        >>> fP, f0 = BMesonBCLFormFactor.make_pair([0.909, -7.11, 66], [0.794, -2.45], mB, mD, mBStar, float("inf"))
        >>> import numpy as np
        >>> q2 = np.linspace(0., (mB - mD)**2, 6)
        >>> resP = fP(q2)
        >>> res0 = f0(q2)
        >>> targetP = np.array([0.74989003, 0.86151387, 1.02142743, 1.24907542, 1.57405952, 2.0433583 ])
        >>> target0 = np.array([0.74989003, 0.75774914, 0.77576395, 0.80549815, 0.84881231, 0.90793663])
        >>> rel_tol = 1e-2
        >>> assert(np.all(np.abs(resP - targetP) < rel_tol * targetP))
        >>> assert(np.all(np.abs(res0 - target0) < rel_tol * target0))
        """
        z0 = z(0., mh, mh_prime)
        NP = len(aP)
        N0 = len(a0)
        if constrained:
            aPN = - (-1.)**NP / NP * sum( (-1)**n * n * aP[n] for n in range(NP) )
        else:
            aPN = 0.
        a0N = ( sum( aP[n] * z0**n for n in range(NP) ) + aPN * z0**NP - sum( a0[n] * z0**n for n in range(N0) ) ) / z0**N0
        fP = cls(aP + [aPN], mh, mh_prime, mpoleP)
        f0 = cls(a0 + [a0N], mh, mh_prime, mpole0)
        return (fP, f0)

class BMesonGenericFF(BMesonFormFactor):
    """
    Form factor for the B → D/K/π or Bs → Ds transitions.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.data.form_factors import BMesonGenericFF
    >>> ff = BMesonGenericFF(5.325, 0.363, -0.78, 1.9, pdata.mass(531), pdata.mass(321))
    >>> import numpy as np
    >>> q2 = np.linspace(0., 1., 6)
    >>> res = ff(q2)
    >>> target = np.array([0.29438108, 0.29666164, 0.29898675, 0.30135757, 0.30377528, 0.30624112])
    >>> rel_tol = 1e-2
    >>> assert(np.all(np.abs(res - target) < rel_tol * target))
    """
    def __init__(self, M_pole, a0, a1, a2, mh, mh_prime):
        super(BMesonGenericFF, self).__init__()
        self.M_pole = M_pole
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.m1 = mh
        self.m2 = mh_prime

    def __call__(self, q2):
        zq2 = z(q2, self.m1, self.m2)
        return (self.a0 + self.a1 * zq2 + self.a2 * zq2**2
                + ((2*self.a2 - self.a1) / 3.) * zq2**3) / (1 - q2 / self.M_pole**2)

class BMesonToPiFF0(BMesonFormFactor):
    """
    Form factor f₀ for the B → π transition.

    >>> from hnlbr.data.particle import ParticleData
    >>> pdata = ParticleData()
    >>> from hnlbr.data.form_factors import BMesonToPiFF0
    >>> ff = BMesonToPiFF0(0.507, -1.77, 1.3, 4, pdata.mass(511), pdata.mass(211))
    >>> import numpy as np
    >>> q2 = np.linspace(0., (pdata.mass(511) - pdata.mass(211))**2, 6)
    >>> res = ff(q2)
    >>> target = np.array([0.2011579 , 0.21556197, 0.25939294, 0.35673317, 0.56227275, 1.01625784])
    >>> rel_tol = 1e-1
    >>> assert(np.all(np.abs(res - target) < rel_tol * np.abs(target)))
    """
    def __init__(self, b0, b1, b2, b3, mh, mh_prime):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.m1 = mh
        self.m2 = mh_prime

    def __call__(self, q2):
        zq2 = z(q2, self.m1, self.m2)
        return self.b0 + self.b1 * zq2 + self.b2 * zq2**2 + self.b3 * zq2**3

class FormFactorDatabase(object):
    """
    Contains all the form factors for the processes of interest.
    """
    def __init__(self, particle_data):
        all_ff = KaonFormFactor.form_factors(particle_data).copy()
        all_ff.update(DMesonFormFactor.form_factors(particle_data))
        all_ff.update(DsMesonFormFactor.form_factors(particle_data))
        # all_ff.update(BMesonFormFactor.new_form_factors(particle_data))
        # all_ff.update(BMesonFormFactor.old_form_factors(particle_data))
        all_ff.update(BMesonFormFactor.best_form_factors(particle_data))
        self.ff = all_ff
        self.pdata = particle_data

    def __getitem__(self, idx):
        h, h_prime = idx
        hc = self.pdata.antiparticle(h)
        hc_prime = self.pdata.antiparticle(h_prime)
        if (h, h_prime) in self.ff:
            return self.ff[(h, h_prime)]
        elif (hc, hc_prime) in self.ff:
            return self.ff[(hc, hc_prime)]
        else:
            raise ValueError("Form factor for " + str(h) + " -> " + str(h_prime) + " not found")

class VectorFormFactorDatabase(FormFactor):
    """
    Class containing the form factors for H → H' transitions, where H' is a
    vector meson.
    """
    def __init__(self, particle_data):
        self.pdata = particle_data
        self._vcoef  = VectorFormFactorDatabase._vector_coefficients()
        self._mclass = VectorFormFactorDatabase._meson_classes()

    @staticmethod
    def _vector_coefficients():
        """
        Coefficients used to express the dimensionless form factors.
        """
        return {
            # (h ,   h' ): [fV   , fA0  , fA1  , fA2  ,  σV   ,  σA0  , σA1  , σA2 , ξV   , ξA0  , ξA1  , ξA2   , MP   , MV   ]
            ("D" , "K*" ): [1.03 , 0.76 , 0.66 , 0.49 ,  0.27 ,  0.17 , 0.30 , 0.67, 0    , 0    , 0.20 ,  0.16 , 1.969, 2.112],
            ("B" , "D*" ): [0.76 , 0.69 , 0.66 , 0.62 ,  0.57 ,  0.59 , 0.78 , 1.40, 0    , 0    , 0    ,  0.41 , 6.275, 6.331],
            ("B" , "rho"): [0.295, 0.231, 0.269, 0.282,  0.875,  0.796, 0.54 , 1.34, 0    , 0.055, 0    , -0.21 , 5.279, 5.325],
            ("Bs", "Ds*"): [0.95 , 0.67 , 0.70 , 0.75 ,  0.372,  0.350, 0.463, 1.04, 0.561, 0.600, 0.510,  0.070, 6.275, 6.331],
            ("Bs", "K*" ): [0.291, 0.289, 0.287, 0.286, -0.516, -0.383, 0    , 1.05, 2.10 , 1.58 , 1.06 , -0.074, 5.367, 5.415]
        }

    @staticmethod
    def _meson_classes():
        """
        Dictionary containing, for each of the meson classes for which the form
        factors are defined, the PDG IDs of the particles belonging to this class.

        Note: Negative PDG codes (antiparticles) are implied.
        """
        return {
            "D"  : [411, 421],
            "B"  : [511, 521],
            "Bs" : [531],
            "rho": [113, 213],
            "K*" : [313, 323],
            "D*" : [413, 423],
            "Ds*": [433]
        }

    def _get_class(self, hadron_id):
        for (k, v) in viewitems(self._mclass):
            if abs(hadron_id) in v:
                return k
        raise ValueError("Hadron " + str(hadron_id) + " not found")

    def make_dimensionless_ff(self, h, h1):
        hc  = self._get_class(h)
        hc1 = self._get_class(h1)
        coef = self._vcoef[(hc, hc1)]
        fV, fA0, fA1, fA2, sV, sA0, sA1, sA2, xV, xA0, xA1, xA2, MP, MV = coef
        def V(q2):
            return fV  / ( (1 - q2/MV**2) * (1 -  sV*q2/MV**2 -  xV*(q2)**2/MV**4) )
        def A0(q2):
            return fA0 / ( (1 - q2/MP**2) * (1 - sA0*q2/MV**2 - xA0*(q2)**2/MV**4) )
        def A1(q2):
            return fA1 / (1 - sA1*q2/MV**2 - xA1*(q2)**2/MV**4)
        def A2(q2):
            return fA2 / (1 - sA2*q2/MV**2 - xA2*(q2)**2/MV**4)
        return V, A0, A1, A2

    def make_dimensionful_ff(self, h, h1):
        V, A0, A1, A2 = self.make_dimensionless_ff(h, h1)
        mh  = self.pdata.mass(h)
        mh1 = self.pdata.mass(h1)
        S = mh + mh1
        D = mh - mh1
        def f(q2):
            return S * A1(q2)
        def g(q2):
            return V(q2) / S
        def aP(q2):
            return -A2(q2) / S
        def aM(q2):
            return ( (S-D)*A0(q2) - S*A1(q2) + D*A2(q2) ) / q2
        return f, g, aP, aM

    def __getitem__(self, idx):
        h, h1 = idx
        return self.make_dimensionful_ff(h, h1)
