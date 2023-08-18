from typing import List, Tuple, Dict, Any, Literal, Final

from .graph.base import BaseGraph, VT, ET
from .graph.scalar import Scalar
from .utils import VertexType, FractionLike

class EGraphVertexType:
    """Type of a vertex in the egraph."""
    Type = Literal[0,1,2,3,4]
    CONCRETE: Final = 0
    GRAPH_S: Final = 1
    GRAPH_T: Final = 2
    EQUIV_S: Final = 3
    EQUIV_T: Final = 4

class SpiderData():
    def __init__(self, ty: VertexType, phase: FractionLike, ninput: int, noutput: int) -> None:
        self.ty = ty
        self.phase = phase
        self.ninput = ninput
        self.noutput = noutput

    def copy(self) -> 'SpiderData':
        return SpiderData(self.ty, self.phase, self.ninput, self.noutput)

    def __repr__(self):
        return f"Spider({self.ty}, {self.phase}, {self.ninput}, {self.noutput})"
        
class EGraph():
    def __init__(self) -> None:
        self.graph: Dict[int,List[int]]                 = dict()
        self._cmap: Dict[int, SpiderData]               = dict()
        self._vindex: int                               = 0
        self.nedges: int                                = 0
        self.ty: Dict[int,EGraphVertexType.Type]        = dict()
        self._vdata: Dict[int,Any]                      = dict()
        self.scalars: Dict[int, Scalar]                 = dict()

    def clone(self) -> 'EGraph':
        cpy = EGraph()
        for v, d in self.graph.items():
            cpy.graph[v] = d.copy()

        cpy._cmap = self._cmap.copy()
        cpy._vindex = self._vindex
        cpy.nedges = self.nedges
        cpy.ty = self.ty.copy()
        cpy._phase = self._phase.copy()
        cpy._vdata = self._vdata.copy()
        cpy.scalars = self.scalars.copy()
        return cpy

    def from_diagram(g: BaseGraph[VT,ET]) -> 'EGraph':
        egraph = EGraph()
        egraph.graph = {v:set(g.graph[v].keys()) for v in g.graph.keys()}

        inputGraph = {v:set() for v in g.graph.keys()}
        for v in g.graph.keys():
            for u in g.graph[v].keys():
                inputGraph[u].add(v)

        egraph._cmap = {vertex:SpiderData(g.ty[vertex], g._phase[vertex], len(inputGraph[vertex]), len(g.graph[vertex])) for vertex in g.graph.keys()}
        egraph._vindex = g._vindex
        egraph.nedges = g.nedges
        egraph.ty = {key:EGraphVertexType.CONCRETE for key in g.ty.keys()}
        egraph._vdata = g._vdata

        g_s = egraph.createNode(EGraphVertexType.GRAPH_S)
        g_t = egraph.createNode(EGraphVertexType.GRAPH_T)
        for i in g._inputs:
            egraph.addEdge(g_s, i)
        for o in g._outputs:
            egraph.addEdge(o, g_t)
        egraph.scalars[g_s] = g.scalar

        return egraph

    def createNode(self, ty, sdata=None, vdata=None) -> int:
        self.graph[self._vindex] = set()
        self.ty[self._vindex] = ty
        if vdata is not None:
            self._vdata[self._vindex] = vdata
        if sdata is None:
            self._cmap[self._vindex] = sdata
        self._vindex += 1

        return self._vindex - 1

    def addEdge(self, u, v) -> None:
        self.graph[u].add(v)
        self.nedges += 1

    def getNeighbors(self, vertex:int) -> List[int]:
        egraph_neighbors = self.graph[vertex]
        neighbors = []
        for u in egraph_neighbors:
            if self.ty(u) == EGraphVertexType.CONCRETE:
                neighbors.append(set([u]))
            elif self.ty(u) == EGraphVertexType.GRAPH_S:
                uneighbors = self.getNeighbors(u)
                neighbors += uneighbors
            elif self.ty(u) == EGraphVertexType.EQUIV_S:
                uneighbors = self.getNeighbors(u)
                neighbors.append(set(uneighbors))

                


