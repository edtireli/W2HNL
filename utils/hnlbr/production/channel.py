# -*- coding: utf-8 -*-

from __future__ import absolute_import

from ..util.misc import hnlid

class Channel(object):
    """
    Abstract class representing a decay channel in which a HNL is produced
    """
    def __init__(self, parent, children, particle_data):
        """
        Initialize process with PDG ID of parent particle and IDs of children particles.
        """
        self.parent = parent
        if len(children) >= 1:
            self.children = children + [hnlid()]
        else:
            raise ValueError("At least one child other than the HNL must be specified")
        self.pdata = particle_data

    def decay_rate(self, hnl_mass, couplings=None):
        """
        Returns the corresponding decay rate for the given HNL mass and couplings.

        Note: it is expected from subclasses to implement this method such that
        a Numpy array can be passed as `hnl_mass`.
        """
        raise RuntimeError("Must be overridden by subclass")

    def branching_ratio(self, hnl_mass, couplings=None):
        """
        Returns the branching ratio of the process, for the given HNL mass and couplings.
        """
        return self.decay_rate(hnl_mass, couplings) / self.pdata.width(self.parent)

    def __str__(self):
        """
        String representation of the decay channel, i.e. A -> B C D...

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> from hnlbr.production.channel import Channel
        >>> ch = Channel(411, [-13], pdata)
        >>> str(ch)
        '411 -> -13 9900015'
        """
        return str(self.parent) + " -> " + " ".join(map(str, self.children))

    def __repr__(self):
        return str(self)

    def _get_lepton_coupling(self, lepton, couplings):
        if couplings is not None:
            alpha = self.pdata.lepton_flavor(lepton, to_str=False) # Flavor ranges from 1 to 3
            U_alpha = couplings[alpha-1] # Index ranges from 0 to 2
        else:
            U_alpha = 1.
        return U_alpha

class ChannelList(object):
    """
    Interface of the class used to generate the list of channels corresponding
    to a given group of processes (e.g. leptonic, semileptonic, ...).
    """
    def __init__(self, particle_data):
        self.pdata = particle_data

    def list_channels(self):
        """
        Returns the list of decay channels corresponding to this process class.
        """
        raise RuntimeError("Must be overridden by subclass")

    def _add_antiparticles(self, particle_list):
        """
        Given a list of particles / antiparticles, add all the respective
        antiparticles / particles if they are missing.

        >>> from hnlbr.data.particle import ParticleData
        >>> pdata = ParticleData()
        >>> from hnlbr.production.channel import ChannelList
        >>> chl = ChannelList(pdata)
        >>> particles = [2212, 111, -211, 211, -521, 441]
        >>> all_particles = chl._add_antiparticles(particles)
        >>> all_particles.sort()
        >>> all_particles
        [-2212, -521, -211, 111, 211, 441, 521, 2212]
        """
        selfconj = list(filter(self.pdata.is_self_conjugate, particle_list))
        charged = list(filter(lambda p: not self.pdata.is_self_conjugate(p), particle_list))
        # Take absolute value, convert to a set to remove duplicates and convert
        # the result back to a list.
        particles = list(set(map(abs, charged)))
        antiparticles = list(map(lambda p: -p, particles))
        all_particles = selfconj + particles + antiparticles
        return all_particles

    def _add_conjugated_channels(self, process_list):
        """
        Warning: partial implementation: may result in duplicated channels.
        """
        conjugated = [[self.pdata.antiparticle(p) for p in proc] for proc in process_list]
        return process_list + conjugated
