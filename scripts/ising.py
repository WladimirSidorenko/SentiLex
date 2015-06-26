#!/usr/bin/env python

"""
Monte Carlo simulation of the 2D Ising model
"""

##################################################################
# Imports
from scipy import *
from scipy import weave
from pylab import *

##################################################################
# Variables and Constants
Nitt = 1000000  # total number of Monte Carlo steps
N = 10          # linear dimension of the lattice, lattice-size= N x N
warm = 1000     # Number of warmup steps
measure = 100   # How often to take a measurement

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

    """

    def __init__(self, a_node_wght = 1., a_edge_wght = 1.):
        """
        Class constructor

        @param a_node_wght - default weight to be used for nodes
        @param a_edge_wght - default weight to be used for edges
        """
        self.dflt_node_wght = a_node_wght
        self.dflt_edge_wght = a_edge_wght
        self.item2nid = dict()
        self.nodes = []
        self.n_nodes = 0

    def add_node(self, a_item, a_weight = None):
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
        self.nodes.append((a_item, a_weight, {}))
        self.item2node[a_item] = self.n_nodes
        self.n_nodes += 1

    def add_edge(self, a_item1, a_item2, a_wght = None, a_allow_self_links = False):
        """
        Add an undirected link between two existing nodes in the ising spin model

        @param a_item1 - first item which should be connected via a link
        @param a_item2 - second item which should be connected via a link
        @param a_wght - initial weight associated with the edge
        @param a_allow_self_links - boolean indicating whether cyclic links to
                      the same node should be allowed

        @return \c void
        """
        if not a_allow_self_links and a_item1 == a_item2:
            return
        assert a_item1 in self.item2nid, "Item '{:s}' not found".format(repr(a_item1))
        assert a_item2 in self.item2nid, "Item '{:s}' not found".format(repr(a_item2))
        inid1 = self.item2nid[a_item1]; inid2 = self.item2nid[a_item2]
        self.nodes[inid1][-1][inid2] = self.nodes[inid2][-1][inid1] = a_wght

##################################################################
# Methods
def CEnergy(latt):
    "Energy of a 2D Ising lattice at particular configuration"
    Ene = 0
    for i in range(len(latt)):
        for j in range(len(latt)):
            S = latt[i,j]
            WF = latt[(i+1)%N, j] + latt[i,(j+1)%N] + latt[(i-1)%N,j] + latt[i,(j-1)%N]
            Ene += -WF*S # Each neighbor gives energy 1.0
    return Ene/2. # Each par counted twice

def RandomL(N):
    "Radom lattice, corresponding to infinite temerature"
    latt = zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            latt[i,j] = sign(2*rand()-1)
    return latt

def SamplePython(Nitt, latt, PW):
    "Monte Carlo sampling for the Ising model in Pythons"
    Ene = CEnergy(latt)         # Starting energy
    Mn = sum(latt)              # Starting magnetization

    Naver=0                     # Measurements
    Eaver=0.0
    Maver=0.0

    N2 = N*N
    for itt in range(Nitt):
        t = int(rand()*N2)
        (i,j) = (t % N, t/N)
        S = latt[i,j]
        WF = latt[(i+1)%N, j] + latt[i,(j+1)%N] + latt[(i-1)%N,j] + latt[i,(j-1)%N]
        P = PW[4+S*WF]
        if P>rand(): # flip the spin
            latt[i,j] = -S
            Ene += 2*S*WF
            Mn -= 2*S

        if itt>warm and itt%measure==0:
            Naver += 1
            Eaver += Ene
            Maver += Mn

    return (Maver/Naver, Eaver/Naver)


def SampleCPP(Nitt, latt, PW, T):
    "The same Monte Carlo sampling in C++"
    Ene = float(CEnergy(latt))  # Starting energy
    Mn = float(sum(latt))       # Starting magnetization

    # Measurements
    aver = zeros(5,dtype=float) # contains: [Naver, Eaver, Maver]

    code="""
    using namespace std;
    int N2 = N*N;
    for (int itt=0; itt<Nitt; itt++){
        int t = static_cast<int>(drand48()*N2);
        int i = t % N;
        int j = t / N;
        int S = latt(i,j);
        int WF = latt((i+1)%N, j) + latt(i,(j+1)%N) + latt((i-1+N)%N,j) + latt(i,(j-1+N)%N);
        double P = PW(4+S*WF);
        if (P > drand48()){ // flip the spin
            latt(i,j) = -S;
            Ene += 2*S*WF;
            Mn -= 2*S;
        }
        if (itt>warm && itt%measure==0){
            aver(0) += 1;
            aver(1) += Ene;
            aver(2) += Mn;
            aver(3) += Ene*Ene;
            aver(4) += Mn*Mn;
        }
    }
    """
    weave.inline(code, ['Nitt','latt','N','PW','Ene','Mn','warm', 'measure', 'aver'],
                 type_converters = weave.converters.blitz, compiler = 'gcc')
    aE = aver[1]/aver[0]
    aM = aver[2]/aver[0]
    cv = (aver[3]/aver[0]-(aver[1]/aver[0])**2)/T**2
    chi = (aver[4]/aver[0]-(aver[2]/aver[0])**2)/T
    return (aM, aE, cv, chi)


if __name__ == '__main__':
    latt = RandomL(N)
    PW = zeros(9, dtype=float)

    wT = linspace(4,0.5,100)
    wMag=[]
    wEne=[]
    wCv=[]
    wChi=[]
    for T in wT:
        # Precomputed exponents
        PW[4+4] = exp(-4.*2/T)
        PW[4+2] = exp(-2.*2/T)
        PW[4+0] = exp(0.*2/T)
        PW[4-2] = exp( 2.*2/T)
        PW[4-4] = exp( 4.*2/T)

        #(maver, eaver) = SamplePython(Nitt, latt, PW)
        (aM, aE, cv, chi) = SampleCPP(Nitt, latt, PW, T)
        wMag.append( aM/(N*N) )
        wEne.append( aE/(N*N) )
        wCv.append( cv/(N*N) )
        wChi.append( chi/(N*N) )

        print T, aM/(N*N), aE/(N*N), cv/(N*N), chi/(N*N)

    plot(wT, wEne, label='E(T)')
    plot(wT, wCv, label='cv(T)')
    plot(wT, wMag, label='M(T)')
    xlabel('T')
    legend(loc='best')
    show()
    plot(wT, wChi, label='chi(T)')
    xlabel('T')
    legend(loc='best')
    show()
