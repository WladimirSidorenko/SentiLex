#!/usr/bin/env python

"""Implementation of the Ising spin model.

"""

##################################################################
# Imports
from __future__ import print_function

from collections import defaultdict
from scipy import *
from pylab import *

import math
# import numpy
import sys

##################################################################
# Variables and Constants
ITEM_IDX = 0
WGHT_IDX = 1
# now unused
# PREV_WGHT_IDX = 2
HAS_FXD_WGHT = 2
FXD_WGHT_IDX = 3
EDGE_IDX = 4

INFINITY = float("inf")
ALPHA = 10
# BETA_RANGE = numpy.linspace(start = 0.1, stop = 1., num = 10)
BETA_RANGE = [0.8]
DFLT_EPSILON = 10 ** -3
MAX_CNT = 5 * 10 ** 3
SPIN_DOMAIN = (-1., 1.)
MAX_EDGE_WGHT = 4

MAX_I = sys.float_info.max
MAX_LOG_I = math.log(MAX_I - 100 if MAX_I > 100 else MAX_I)


##################################################################
# Class
class Ising(object):
    """
    Class representing the Ising spin model

    Instance variables:
    nodes - list of nodes in the model
    n_nodes - number of nodes stored in the model
    item2nid - dictionary mapping items to their respective node id's
    dflt_node_wght - default weight to be used for nodes
    dflt_edge_wght - default weight to be used for edges

    Public methods:
    add_node - add node to the ising spin model
    add_edge - connect two nodes via an undirected link
    reweight - re-estimate weights of undirected links
    """

    def __init__(self, a_node_wght=0., a_edge_wght=1.):
        """
        Class constructor

        @param a_node_wght - default weight to be used for nodes
        @param a_edge_wght - default weight to be used for edges
        """
        ## default weight to be used for nodes
        self.dflt_node_wght = a_node_wght
        ## default weight to be used for edges
        self.dflt_edge_wght = a_edge_wght
        ## dictionary mapping items to their respective node id's
        self.item2nid = dict()
        ## list of nodes in the model
        self.nodes = []
        ## number of nodes stored in the model
        self.n_nodes = 0
        ## external temperature
        self.beta = -1

    def __contains__(self, a_item):
        """
        Check if an item is present among the Ising nodes

        @param a_node_wght - default weight to be used for nodes

        @return \c True
        """
        return (a_item in self.item2nid)

    def __getitem__(self, a_item):
        """
        Return node corresponding to the given item

        @param a_item - item which should be retrieved

        @return node corresponding to the given item
        """
        return self.nodes[self.item2nid[a_item]]

    def __setitem__(self, a_item, a_value):
        """
        Return node corresponding to the given item

        @param a_item - item which should be retrieved

        @return node corresponding to the given item
        """
        ret = self.nodes[self.item2nid[a_item]] = a_value
        return ret

    def add_node(self, a_item, a_weight=None):
        """
        Add node to the ising spin model

        @param a_item - item corresponding to the new node
        @param a_weight - initial weight associated with the new node

        @return \c void
        """
        if a_item in self.item2nid:
            return
        if a_weight is None:
            a_weight = self.dflt_node_wght
        # each node has the form: (item, wght, has_fxd_wght, fxd_wght, edges:
        # {trg: edge_weight})

        self.nodes.append([a_item, a_weight,
                           0 if a_weight is None else 1.,
                           a_weight, defaultdict(lambda: 0.)])
        self.item2nid[a_item] = self.n_nodes
        self.n_nodes += 1

    def add_edge(self, a_item1, a_item2, a_wght, a_allow_self_links=False,
                 a_add_missing=False):
        """Connect two nodes via an undirected link

        @param a_item1 - first item which should be connected via a link
        @param a_item2 - second item which should be connected via a link
        @param a_wght - initial weight associated with the edge
        @param a_allow_self_links - boolean indicating whether cyclic links to
                      the same node should be allowed

        @param a_add_missing - boolean flag indicating whether missing nodes
                      should be added

        @return \c void

        """
        if not a_allow_self_links and a_item1 == a_item2:
            return
        if a_item1 not in self.item2nid:
            if a_add_missing:
                self.add_node(a_item1)
            else:
                raise RuntimeError(
                    "Item '{:s}' not found in Ising model".format(
                        repr(a_item1)))
        if a_item2 not in self.item2nid:
            if a_add_missing:
                self.add_node(a_item2)
            else:
                raise RuntimeError(
                    "Item '{:s}' not found in Ising model".format(
                        repr(a_item2)))
        inid1 = self.item2nid[a_item1]
        inid2 = self.item2nid[a_item2]
        new_wght = self.nodes[inid1][EDGE_IDX][inid2] + a_wght
        if new_wght < MAX_EDGE_WGHT:
            self.nodes[inid1][EDGE_IDX][inid2] = new_wght
        new_wght = self.nodes[inid2][EDGE_IDX][inid2] + a_wght
        if new_wght < MAX_EDGE_WGHT:
            self.nodes[inid2][EDGE_IDX][inid1] = new_wght

    def iteritems(self):
        """Return iterator over nodes.

        @return iterator

        @raise StopIteration

        """
        if not self.nodes:
            raise StopIteration
        for inode in self.nodes:
            yield (inode[0], inode)

    def reweight(self):
        """
        Re-estimate weights of undirected links by multiplying them with
        `1/sqrt(d(i) * d(j))` where `d(i)` and `d(j)` denote the degree of the
        adjacent nodes.

        @return \c void
        """
        children = []
        src_degree = 0.
        trg_degree = 0.
        for isrc_nid in xrange(self.n_nodes):
            children = self.nodes[isrc_nid][EDGE_IDX]
            src_degree = math.sqrt(float(len(children)))
            for itrg_nid in children:
                # multiply edge weight with 1/sqrt(d(i) * d(j))
                trg_degree = math.sqrt(
                    float(len(self.nodes[itrg_nid][EDGE_IDX])))
                children[itrg_nid] *= 1. / (src_degree * trg_degree)

    def train(self, a_betas=BETA_RANGE, a_epsilon=DFLT_EPSILON, a_plot=None):
        """Determine spin orientation of the model

        @param a_beta - range of beta values to test
        @param a_epsilon - epsilon value to determine convergence
        @param a_plot - boolean flag indicating whether the energy changes
          should be plotted

        @return \c void

        """
        beta2em = dict()
        ibeta = best_beta = -1
        energy = magn = min_magn = INFINITY
        # iterate over all specified beta values
        for i, ibeta in enumerate(a_betas):
            print("Iteration #{:d}: beta = {:f}".format(i, ibeta),
                  file=sys.stderr)
            # optimize spin orientation of the model
            energy, magn = self._train(ibeta, a_epsilon)
            # print("Energy = {:f}, magnetization = {:f}".format(energy, magn),
            # file=sys.stderr)
            beta2em[ibeta] = (energy, magn)
            if not math.isnan(magn) and magn < min_magn:
                min_magn = magn
                best_beta = ibeta
        # re-train the model with the best parameter setting
        if best_beta != ibeta:
            print("Final iteration: beta = {:f}".format(best_beta),
                  file=sys.stderr)
            self.beta = best_beta
            self._train(best_beta, a_epsilon)
        # plot energy/magnetization development, if asked to do so
        if a_plot is not None:
            self._plot(a_plot, beta2em)

    def _train(self, a_beta=None, a_epsilon=DFLT_EPSILON):
        """Helper function for doing single training run with the given beta

        @param a_beta - beta value to use
        @param a_epsilon - epsilon value to determine convergence

        @return 2-tuple holding energy and magnetization values

        """
        if a_beta is None:
            a_beta = self.beta
        cnt = 0
        energy = magn = 0.
        a_epsilon = abs(a_epsilon)
        prev_energy = prev_magn = INFINITY
        # set initial weights
        self._train_init(a_beta)
        while (prev_energy == INFINITY
               or abs(prev_energy - energy) > a_epsilon) \
                and cnt < MAX_CNT:
            for inode in self.nodes:
                # update node's spin orientation (according to Takamura's code,
                # we do the update relying on the new spin weights)
                # was `a_idx = PREV_WGHT_IDX`
                inode[WGHT_IDX] = self._compute_mean(inode,
                                                     a_idx=WGHT_IDX,
                                                     a_beta=a_beta)
                assert not math.isnan(inode[WGHT_IDX]), \
                    "Infinite weight on node '{:s}'".format(
                        repr(inode[ITEM_IDX]))
            # re-estimate energy and magnetization
            prev_energy, prev_magn = energy, magn
            energy, magn = self._measure(a_beta=a_beta)
            print("Run #{:d}: energy = {:f}, magnetization = {:f}".format(
                cnt, energy, magn), file=sys.stderr)
            # after all nodes have been processed, remember the newly computed
            # node weights as the old ones (commented out according to
            # Takamura's code)
            # for inode in self.nodes:
            #     inode[PREV_WGHT_IDX] = inode[WGHT_IDX]

            # prevent inifinite loops
            if prev_magn == INFINITY and cnt > 10:
                break
            cnt += 1
        return (energy, magn)

    def _train_init(self, a_beta):
        """
        Helper function for initializing state weights

        @param a_beta - beta value to use

        @return \c void
        """
        for inode in self.nodes:
            if inode[HAS_FXD_WGHT]:
                inode[WGHT_IDX] = inode[FXD_WGHT_IDX]
            else:
                inode[WGHT_IDX] = 0.

    def _measure(self, a_beta=None):
        """Measure energy and magnetization of the current model

        @param a_beta - beta value to use

        @return 2-tuple holding energy and magnetization

        """
        if a_beta is None:
            a_beta = self.beta
        probs = []
        edge_wght = 0.
        energy = magn = sum1 = sum2 = sum3 = node_wght = 0.
        for inode in self.nodes:
            node_wght = inode[WGHT_IDX]
            edge_wght = sum([self.nodes[k][WGHT_IDX] * v
                             for k, v in inode[EDGE_IDX].iteritems()])
            probs = self._compute_probs(inode, WGHT_IDX, a_beta)

            magn += node_wght  # raw spin orientation
            # magn += sum([x_i * iprob
            # for x_i, iprob in zip(SPIN_DOMAIN, probs)]) # model expectation

            sum1 -= node_wght * edge_wght
            # entropy
            sum2 -= sum([iprob * math.log(iprob, 2.)
                         for iprob in probs if iprob])
            # penalty
            if inode[HAS_FXD_WGHT]:
                sum3 += sum([iprob * (x_i - inode[FXD_WGHT_IDX])**2
                             for iprob, x_i in zip(probs, SPIN_DOMAIN)])
            # mean = self._compute_mean(inode)
            # # mean = inode[WGHT_IDX]
            # q_pos = (1. + mean) / 2.; q_neg = (1. - mean) / 2.
            # lq_pos = log(q_pos) if q_pos else 0.
            # lq_neg = log(q_neg) if q_neg else 0.
            # energy -= (self.beta / 2.) * inode[WGHT_IDX] * \
            #     sum([self.nodes[k][WGHT_IDX] * v
            #     for k, v in inode[EDGE_IDX].iteritems()]) - \
            #     q_pos * lq_pos - q_neg * lq_neg
        # print("sum1 =", repr(sum1), file = sys.stderr)
        # print("sum2 =", repr(sum2), file = sys.stderr)
        # print("sum3 =", repr(sum3), file = sys.stderr)
        energy = (sum1 * a_beta / 2.) + sum2 + sum3
        return (energy, magn / float(self.n_nodes))

    def _compute_mean(self, a_node, a_idx=WGHT_IDX, a_beta=None):
        """
        Compute mean of spin orientation of the given node

        @param a_node - node whose spin orientation should be computed
        @param a_idx - list index of neighbor weights
        @param a_beta - beta value to use

        @return float representing the mean spin orientation
        """
        probs = self._compute_probs(a_node, a_idx, a_beta)
        # print("probs {:s} = ".format(
        # repr(a_node[0])), repr(probs), file=sys.stderr)
        return sum([x_i * iprob for x_i, iprob in zip(SPIN_DOMAIN, probs)])

    def _compute_probs(self, a_node, a_idx=WGHT_IDX, a_beta=None):
        """
        Compute probabilities of spin orientations for the given node

        @param a_node - node whose spin orientation should be computed
        @param a_idx - list index of neighbor weights
        @param a_beta - beta value to use

        @return vector representing probabilities of each value in SPIN_DOMAIN
        """
        if a_beta is None:
            a_beta = self.beta

        # print("beta =", repr(a_beta), file = sys.stderr)
        # print("edges =", repr([(self.nodes[k][a_idx], v)
        # for k, v in a_node[EDGE_IDX].iteritems()]), file = sys.stderr)
        # print("sum =", repr(sum([self.nodes[k][a_idx] * v
        # for k, v in a_node[EDGE_IDX].iteritems()])), file = sys.stderr)

        edge_wght = a_beta * sum([self.nodes[k][a_idx] * v
                                  for k, v in a_node[EDGE_IDX].iteritems()])
        # print("edge_wght =", repr(edge_wght), file = sys.stderr)
        # prevent overflow
        probs = [x_i * edge_wght - ALPHA * a_node[HAS_FXD_WGHT]
                 * ((x_i - a_node[FXD_WGHT_IDX]) ** 2)
                 for x_i in SPIN_DOMAIN]
        probs = [exp(x_i) if x_i < MAX_LOG_I else MAX_I for x_i in probs]
        norm = 0.
        for iprob in probs:
            if MAX_I - norm > iprob:
                norm += iprob
            else:
                norm = MAX_I
                break
        # print("probs =", repr(probs), file = sys.stderr)
        # print("norm =", repr(norm), file = sys.stderr)
        if norm:
            return [iprob / norm for iprob in probs]
        return probs

    def _plot(self, a_plot, a_beta2em):
        """Plot the development of energy/magnetization

        @param a_plot - extension of the file in which new plot should be saved
        @param a_beta2em - dictionary mapping beta vaules to free
        energy/magnetization

        @return \c void

        """
        betas = []
        energy = []
        magnetization = []
        for ibeta, (ienergy, imagnet) in a_beta2em.iteritems():
            betas.append(ibeta)
            energy.append(ienergy)
            magnetization.append(imagnet)
        figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
        rc("text", usetex=True)
        plot(betas, energy, label=r"E($\beta$)")
        xlabel(r"$\beta$")
        savefig("takamura-energy." + a_plot, format=a_plot)
        clf()
        plot(betas, magnetization, label=r"M($\beta$)")
        xlabel(r"$\beta$")
        savefig("takamura-magnetization." + a_plot, format=a_plot)

##################################################################
# Reference Implementation
# Nitt = 1000000  # total number of Monte Carlo steps
# N = 10          # linear dimension of the lattice, lattice-size= N x N
# warm = 1000     # Number of warmup steps
# measure = 100   # How often to take a measurement

# def CEnergy(latt):
#     "Energy of a 2D Ising lattice at particular configuration"
#     Ene = 0
#     for i in range(len(latt)):
#         for j in range(len(latt)):
#             S = latt[i,j]
#             WF = latt[(i+1)%N, j] + latt[i,(j+1)%N] \
#                  + latt[(i-1)%N,j] + latt[i,(j-1)%N]
#             Ene += -WF*S # Each neighbor gives energy 1.0
#     return Ene/2. # Each par counted twice

# def RandomL(N):
#     "Radom lattice, corresponding to infinite temerature"
#     latt = zeros((N,N), dtype=int)
#     for i in range(N):
#         for j in range(N):
#             latt[i,j] = sign(2*rand()-1)
#     return latt

# def SamplePython(Nitt, latt, PW):
#     "Monte Carlo sampling for the Ising model in Pythons"
#     Ene = CEnergy(latt)         # Starting energy
#     Mn = sum(latt)              # Starting magnetization

#     Naver=0                     # Measurements
#     Eaver=0.0
#     Maver=0.0

#     N2 = N*N
#     for itt in range(Nitt):
#         t = int(rand()*N2)
#         (i,j) = (t % N, t/N)
#         S = latt[i,j]
#         WF = latt[(i+1)%N, j] + latt[i,(j+1)%N] \
#              + latt[(i-1)%N,j] + latt[i,(j-1)%N]
#         P = PW[4+S*WF]
#         if P>rand(): # flip the spin
#             latt[i,j] = -S
#             Ene += 2*S*WF
#             Mn -= 2*S

#         if itt>warm and itt%measure==0:
#             Naver += 1
#             Eaver += Ene
#             Maver += Mn

#     return (Maver/Naver, Eaver/Naver)


# def SampleCPP(Nitt, latt, PW, T):
#     "The same Monte Carlo sampling in C++"
#     Ene = float(CEnergy(latt))  # Starting energy
#     Mn = float(sum(latt))       # Starting magnetization

#     # Measurements
#     aver = zeros(5,dtype=float) # contains: [Naver, Eaver, Maver]

#     code="""
#     using namespace std;
#     int N2 = N*N;
#     for (int itt=0; itt<Nitt; itt++){
#         int t = static_cast<int>(drand48()*N2);
#         int i = t % N;
#         int j = t / N;
#         int S = latt(i,j);
#         int WF = latt((i+1)%N, j) + latt(i,(j+1)%N) \
#             + latt((i-1+N)%N,j) + latt(i,(j-1+N)%N);
#         double P = PW(4+S*WF);
#         if (P > drand48()){ // flip the spin
#             latt(i,j) = -S;
#             Ene += 2*S*WF;
#             Mn -= 2*S;
#         }
#         if (itt>warm && itt%measure==0){
#             aver(0) += 1;
#             aver(1) += Ene;
#             aver(2) += Mn;
#             aver(3) += Ene*Ene;
#             aver(4) += Mn*Mn;
#         }
#     }
#     """
#     weave.inline(code, ['Nitt','latt','N','PW','Ene',
#     'Mn','warm', 'measure', 'aver'],
#                  type_converters = weave.converters.blitz, compiler = 'gcc')
#     aE = aver[1]/aver[0]
#     aM = aver[2]/aver[0]
#     cv = (aver[3]/aver[0]-(aver[1]/aver[0])**2)/T**2
#     chi = (aver[4]/aver[0]-(aver[2]/aver[0])**2)/T
#     return (aM, aE, cv, chi)


# if __name__ == '__main__':
#     latt = RandomL(N)
#     PW = zeros(9, dtype=float)

#     wT = linspace(4,0.5,100)
#     wMag=[]
#     wEne=[]
#     wCv=[]
#     wChi=[]
#     for T in wT:
#         # Precomputed exponents
#         PW[4+4] = exp(-4.*2/T)
#         PW[4+2] = exp(-2.*2/T)
#         PW[4+0] = exp(0.*2/T)
#         PW[4-2] = exp( 2.*2/T)
#         PW[4-4] = exp( 4.*2/T)

#         #(maver, eaver) = SamplePython(Nitt, latt, PW)
#         (aM, aE, cv, chi) = SampleCPP(Nitt, latt, PW, T)
#         wMag.append( aM/(N*N) )
#         wEne.append( aE/(N*N) )
#         wCv.append( cv/(N*N) )
#         wChi.append( chi/(N*N) )

#         print(T, aM/(N*N), aE/(N*N), cv/(N*N), chi/(N*N))

#     plot(wT, wEne, label='E(T)')
#     plot(wT, wCv, label='cv(T)')
#     plot(wT, wMag, label='M(T)')
#     xlabel('T')
#     legend(loc='best')
#     show()
#     plot(wT, wChi, label='chi(T)')
#     xlabel('T')
#     legend(loc='best')
#     show()
