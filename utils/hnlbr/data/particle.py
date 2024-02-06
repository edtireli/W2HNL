# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import particletools.tables as pt

from .constants import (GF, thetaW)
from .ckm import CKMabs
from .decay_constants import (PseudoscalarDC, VectorDecayConstants)
from ..util.misc import hnlid
from ..util import units as u
from . import form_factors

class ParticleData(object):
    """
    Wrapper class around `particledatatools`.
    """
    def __init__(self):
        self.GF = GF()
        self.thetaW = thetaW()
        self.hnlid = hnlid()
        self.pdg = pt.PYTHIAParticleData()
        self.ckm = CKMabs()
        self.pseudoscalar_decay_constants = PseudoscalarDC()
        self.vector_decay_constants = VectorDecayConstants()
        self._quark_chars = ['d', 'u', 's', 'c', 'b', 't']
        self._lepton_key = {11: 1, 12: 1, 13: 2, 14: 2, 15: 3, 16: 3}
        self._flavor_str = ["e", "mu", "tau"]

        # Override some wrong values
        self._pdg_id_name_overrides = {
            "D*0"     : +423,
            "D*bar0"  : -423,
            "D*+"     : +413,
            "D*-"     : -413,
            "B*0"     : +513,
            "B*bar0"  : -513,
            "B*+"     : +523,
            "B*-"     : -523,
            "B*_s0"   : +533,
            "B*_sbar0": -533
        }
        self._pdg_id_properties_overrides = {
            423: { "mass": 2006.85 * u.MeV() },
            513: { "mass": 5324.65 * u.MeV() },
            523: { "mass": 5324.65 * u.MeV() },
            533: { "mass": 5415.4  * u.MeV() }
        }

        # At the very end because it is self-referential
        self.ffdb = form_factors.FormFactorDatabase(self)
        self.ffdb_vec = form_factors.VectorFormFactorDatabase(self)

    def pdg_id(self, name):
        """
        Returns the PDG ID of the particle.

        This includes some fixes compared to the function from `particletools`.
        """
        if name in self._pdg_id_name_overrides:
            return self._pdg_id_name_overrides[name]
        else:
            return self.pdg.pdg_id(name)

    def mass(self, pdgid):
        """
        Returns the mass of the particle of the corresponding PDG ID.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdg = ParticleData()
        >>> mp = pdg.mass(2212)
        >>> assert(abs(mp - 0.93827) < 0.000005)
        """
        if abs(pdgid) in self._pdg_id_properties_overrides:
            return self._pdg_id_properties_overrides[abs(pdgid)]["mass"]
        else:
            return self.pdg.mass(pdgid)

    def stable(self, pdgid):
        """
        Returns whether the particle is stable or not.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdg = ParticleData()
        >>> pdg.stable(2212)
        True
        """
        return self.ctau_m(pdgid) == float("inf")

    def ctau_m(self, pdgid):
        """
        Returns the lifetime cτ of the particle (in meters).

        >>> from hnlbr.data.particle import ParticleData
        >>> pdg = ParticleData()
        >>> ct_pi0 = pdg.ctau_m(111)
        >>> assert(abs(ct_pi0 - 25.5e-9) < 5 * 0.5e-9)
        >>> ct_pi = pdg.ctau_m(211)
        >>> assert(abs(ct_pi - 7.8045) < 5 * 0.0015)
        """
        return self.pdg.ctau(pdgid) * 1e-2 # Convert from centimeters

    def lifetime_s(self, pdgid):
        """
        Returns the lifetime of the particle (in seconds).

        >>> from hnlbr.data.particle import ParticleData
        >>> pdg = ParticleData()
        >>> t_pi0 = pdg.lifetime_s(111)
        >>> assert(abs(t_pi0 - 8.52e-17) < 5 * 0.18e-17)
        >>> t_pi = pdg.lifetime_s(211)
        >>> assert(abs(t_pi - 2.6033e-8) < 5 * 0.0005e-8)
        """
        return self.ctau_m(pdgid) / u.s_per_m()

    def width(self, pdgid):
        """
        Returns the width of the particle (in GeV).

        >>> from hnlbr.data.particle import ParticleData
        >>> pdg = ParticleData()
        >>> w_H = pdg.width(25)
        >>> assert(w_H < 3.)
        """
        return 1. / (self.ctau_m(pdgid) * u.m())

    def charge_code(self, pdgid):
        """
        Return 3 times the electric charge of the particle.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.charge_code(2212)
        3
        >>> pdata.charge_code(111)
        0
        >>> pdata.charge_code(11)
        -3
        """
        return int(3. * self.charge(pdgid))

    def charge(self, pdgid):
        """
        Return the electric charge of the particle.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.charge(2212)
        1.0
        >>> pdata.charge(111)
        0.0
        >>> pdata.charge(11)
        -1.0
        """
        return self.pdg.charge(pdgid)

    def quark_content(self, pdgid, to_char=True):
        """
        Returns the quark content of a given hadron, based on its PDG ID.

        Particles and anti-particles are treated the same.

        Options:
        * to_char: whether to convert the result to characters, i.e. [2, 2, 1] -> ['u', 'u', 'd']

        Note: only minimal checks are done to ensure that the particle is indeed
        a hadron, so user discretion is advised. This method is not reliable for
        hidden flavored mesons such as the π⁰, since they can be a coherent
        superposition of quark flavor eigenstates.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.quark_content(2212)
        ['u', 'u', 'd']
        >>> pdata.quark_content(2212, to_char=False)
        [2, 2, 1]
        >>> pdata.quark_content(-211)
        ['u', 'd']
        """
        if pdgid == 111: # π⁰
            raise ValueError("Not implemented for π⁰")
        id_str = str(abs(pdgid) % 10000)
        if len(id_str) < 3:
            raise ValueError("Particle " + id_str + " does not look like a hadron")
        # Last digit encodes spin: ignore it
        quark_codes = [int(c) for c in id_str[0:-1]]
        if to_char:
            quark_chars = [self._quark_chars[qc-1] for qc in quark_codes]
            return quark_chars
        else:
            return quark_codes

    def lepton_flavor(self, pdgid, to_str=True):
        """
        Returns the flavor of a given SM lepton, based on its PDG ID.

        Options:
        * to_str: whether to convert the result to string, i.e. 1 -> 'e', 2 -> 'mu', 3 -> 'tau'

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.lepton_flavor(-11)
        'e'
        >>> pdata.lepton_flavor(14)
        'mu'
        >>> pdata.lepton_flavor(-15, to_str=False)
        3
        """
        pdgid = abs(pdgid)
        if not (pdgid >= 11 and pdgid <= 16):
            raise ValueError("Particle " + str(pdgid) + " does not look like a SM lepton")
        flavor_idx = self._lepton_key[pdgid]
        if to_str:
            flavor_str = self._flavor_str[flavor_idx - 1]
            return flavor_str
        else:
            return flavor_idx

    def is_self_conjugate(self, pdgid):
        """
        Returns whether the particle is real / self-conjugate.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.is_self_conjugate(2212)
        False
        >>> pdata.is_self_conjugate(111)
        True
        >>> pdata.is_self_conjugate(-211)
        False
        """
        #FIXME: all our overrides are not self-conjugated
        if pdgid in self._pdg_id_name_overrides.values():
            return False
        p_name = self.pdg.name(abs(pdgid))
        try:
            ap_name = self.pdg.name(-abs(pdgid))
        except KeyError: # Anti-particle is missing -> self-conjugate
            return True
        return False

    def antiparticle(self, pdgid):
        """
        Return the antiparticle, or the particle itself if it is self-conjugated.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> pdata.antiparticle(111)
        111
        >>> pdata.antiparticle(211)
        -211
        >>> pdata.antiparticle(-2212)
        2212
        >>> pdata.antiparticle(311)
        -311
        >>> pdata.antiparticle(113)
        113
        >>> pdata.antiparticle(11)
        -11
        >>> pdata.antiparticle(25)
        25
        """
        if self.is_self_conjugate(pdgid):
            return pdgid
        else:
            return -pdgid
