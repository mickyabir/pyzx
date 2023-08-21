from typing import List, Tuple, Dict, Any, Literal, Final, Set

from .graph.base import BaseGraph, VT, ET
from .graph.scalar import Scalar
from .utils import VertexType, FractionLike

class EGraphVertexType:
    """Type of a vertex in the egraph."""
    Type = Literal[0,1,2,3,4,5]
    CONCRETE: Final = 0
    IDENTITY: Final = 1
    GRAPH: Final  = 2
    EQUIV: Final  = 3

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
        self._gindex: int                               = 0
        self._eindex: int                               = 0
        self.ty: Dict[int,EGraphVertexType.Type]        = dict()
        self._vdata: Dict[int,Any]                      = dict()
        self.scalars: Dict[int, Scalar]                 = dict()
        self.subgraphs: List[Tuple[int]]                = []
        self.equivalences: List[Tuple[int]]             = []

    def clone(self) -> 'EGraph':
        cpy = EGraph()
        for v, d in self.graph.items():
            cpy.graph[v] = d.copy()

        cpy._cmap = self._cmap.copy()
        cpy._vindex = self._vindex
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
        egraph.ty = {key:EGraphVertexType.CONCRETE for key in g.ty.keys()}
        egraph._vdata = g._vdata

        for u, v in g.edge_set():
            inode = egraph.createNode(EGraphVertexType.IDENTITY)
            egraph.removeEdge(u, v)
            egraph.removeEdge(v, u)
            egraph.addEdge(u, inode)
            egraph.addEdge(inode, u)
            egraph.addEdge(v, inode)
            egraph.addEdge(inode, v)

        g_s = egraph.createNode(EGraphVertexType.GRAPH)
        g_t = egraph.createNode(EGraphVertexType.GRAPH)
        egraph.subgraphs.append((g_s, g_t))
        for i in g._inputs:
            egraph.addEdge(g_s, i)
            egraph.addEdge(i, g_s)
        for o in g._outputs:
            egraph.addEdge(g_t, o)
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
        if v not in self.graph[u]:
            self.graph[u].add(v)

    def removeEdge(self, u, v) -> None:
        if v in self.graph[u]:
            self.graph[u].remove(v)


    def getNeighbors(self, vertex:int) -> List[int]:
        return self.graph[vertex]

    def _getNeighborsConcrete(self, vertex:int, visited:Set[int]) -> List[int]:
        egraph_neighbors = self.getNeighbors(vertex)
        neighbors = []
        visited.add(vertex)
        for u in egraph_neighbors:
            if u in visited:
                continue
            if self.ty[u] == EGraphVertexType.CONCRETE:
                neighbors.append(set([u]))
            elif self.ty[u] == EGraphVertexType.GRAPH or self.ty[u] == EGraphVertexType.IDENTITY:
                uneighbors = self._getNeighborsConcrete(u, visited)
                neighbors += uneighbors
            elif self.ty[u] == EGraphVertexType.EQUIV:
                uneighbors = self._getNeighborsConcrete(u, visited)
                neighbors.append(set(uneighbors))

        for v in visited:
            neighbors.remove({v}) if {v} in neighbors else None

        return neighbors

    def getNeighborsConcrete(self, vertex:int, visited:Set[int]=set()) -> List[int]:
        return self._getNeighborsConcrete(vertex, set())
