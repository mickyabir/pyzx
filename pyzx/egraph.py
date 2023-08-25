from typing import List, Tuple, Dict, Any, Literal, Final, Set, Optional, Callable, TypeVar, Union

from .simplify import Stats

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

    def __eq__(self, other):
        return self.ty == other.ty and self.phase == other.phase and self.ninput == other.ninput and self.noutput == other.noutput

    def __hash__(self):
        return hash(f'{self.ty},{self.phase},{self.ninput},{self.noutput}')
        
class EGraph():
    def __init__(self) -> None:
        self.graph: Dict[int,Set[int]]                  = dict()
        self._names: Dict[int, str]                     = dict()

        # Maps vertex number to spider data
        self._cmap: Dict[int, SpiderData]               = dict()
        # Hashes spider to vertex number
        self._spiders: Dict[SpiderData, int]            = dict()

        self.qubits: Dict[int, int]                     = dict()

        self._vindex: int                               = 0
        self._sindex: int                               = 0
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

        cpy.qubits = self.qubits.copy()
        cpy._names = self._names.copy()
        cpy._cmap = self._cmap.copy()
        cpy._vindex = self._vindex
        cpy._sindex = self._sindex
        cpy._gindex = self._gindex
        cpy._eindex = self._eindex
        cpy.ty = self.ty.copy()
        cpy._vdata = self._vdata.copy()
        cpy.scalars = self.scalars.copy()
        cpy.subgraphs = self.subgraphs.copy()
        cpy.equivalences = self.equivalences.copy()
        return cpy

    def from_diagram(g: BaseGraph[VT,ET]) -> 'EGraph':
        egraph = EGraph()

        egraph.graph = {v:set() for v in g.graph.keys()}
        egraph.qubits = {v:g.qubits()[v] for v in g.graph.keys()}
        inputCount = {v:0 for v in g.graph.keys()}
        visited = set()
        candidates = g.inputs().copy()
        while candidates:
            v = candidates.pop(0)
            visited.add(v)
            for u in g.graph[v].keys():
                if u not in visited:
                    egraph.graph[v].add(u)
                    inputCount[u] += 1

                    if u not in candidates:
                        candidates.append(u)

        for v in egraph.graph.keys():
            sdata = SpiderData(g.ty[v], g._phase[v], inputCount[v], len(egraph.graph[v]))
            egraph._cmap[v] = sdata
            egraph.ty[v] = EGraphVertexType.CONCRETE
            if sdata in egraph._spiders.keys():
                egraph._names[v] = egraph._spiders[sdata]
            else:
                egraph._names[v] = f's{v}'
                egraph._spiders[sdata] = f's{v}'

        egraph._vindex = g._vindex
        egraph._sindex = g._vindex

        nodes = list(egraph.graph.keys()).copy()
        for u in nodes:
            neighbors = egraph.graph[u].copy()
            for v in neighbors:
                inode = egraph.createIdentNode()
                egraph.removeEdge(u, v)
                egraph.addEdge(u, inode)
                egraph.addEdge(inode, v)

        g_s, g_t = egraph.createGraphNodes()
        for i in g._inputs:
            egraph.addEdge(g_s, i)
        for o in g._outputs:
            egraph.addEdge(o, g_t)
        egraph.scalars[g_s] = g.scalar

        return egraph

    def concreteNodes(self) -> List[int]:
        return list(self._cmap.keys())

    def createSpiderNode(self, sdata:SpiderData = None, vdata=None) -> int:
        spider = self._vindex
        self.graph[spider] = set()
        self.ty[spider] = EGraphVertexType.CONCRETE
        self._cmap[spider] = sdata

        if sdata in self._spiders.keys():
            self._names[spider] = self._spiders[sdata]
        else:
            self._names[spider] = f's{self._sindex}'
            self._spiders[sdata] = f's{self._sindex}'

        if vdata is not None:
            self._vdata[self._vindex] = vdata

        self._sindex += 1
        self._vindex += 1

        return spider

    def createGraphNodes(self) -> List[int]:
        g_s = self._vindex
        g_t = self._vindex + 1

        self.graph[g_s] = set()
        self.graph[g_t] = set()
        self.ty[g_s] = EGraphVertexType.GRAPH
        self.ty[g_t] = EGraphVertexType.GRAPH
        self.subgraphs.append((g_s, g_t))

        self._names[g_s] = f'g{self._gindex}_s'
        self._names[g_t] = f'g{self._gindex}_t'

        self._gindex += 1
        self._vindex += 2

        return g_s, g_t

    def createEquivNodes(self) -> List[int]:
        e_s = self._vindex
        e_t = self._vindex + 1

        self.graph[e_s] = set()
        self.graph[e_t] = set()
        self.ty[e_s] = EGraphVertexType.EQUIV
        self.ty[e_t] = EGraphVertexType.EQUIV
        self.equivalences.append((e_s, e_t))

        self._names[e_s] = f'e{self._eindex}_s'
        self._names[e_t] = f'e{self._eindex}_t'

        self._eindex += 1
        self._vindex += 2

        return e_s, e_t

    def createIdentNode(self) -> int:
        iden = self._vindex
        self.graph[iden] = set()
        self.ty[iden] = EGraphVertexType.IDENTITY
        self._names[iden] = f'id'
        self._vindex += 1

        return iden

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
                neighbors.append(u)
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

MatchObject = TypeVar('MatchObject')
RewriteOutputType = Tuple[Dict[ET,List[int]], List[VT], List[ET], bool]
MatchBialgType = Tuple[VT,VT,List[VT],List[VT]]
def match_bialg_parallel(
        g: EGraph, 
        matchf:Optional[Callable[[ET],bool]]=None, 
        num: int=-1
        ) -> List[MatchBialgType[VT]]:
    """Finds noninteracting matchings of the bialgebra rule.
    
    :param g: An instance of a ZX-EGraph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
       tries to find as many as possible.
    :rtype: List of 4-tuples ``(v1, v2, neighbors_of_v1,neighbors_of_v2)``
    """
    types = lambda x: g._cmap[x].ty
    phases = lambda x: g._cmap[x].phase
    i = 0
    m = []
    candidates = g.concreteNodes()
    import pdb
    pdb.set_trace()
    while (num == -1 or i < num) and len(candidates) > 0:
        v0 = candidates.pop(0)
        v0_neighbors = g.getNeighborsConcrete(v0)
        for v1 in v0_neighbors:
            v0t = types(v0)
            v1t = types(v1)
            v0p = phases(v0)
            v1p = phases(v1)

            if (v0p == 0 and v1p == 0 and
            ((v0t == VertexType.Z and v1t == VertexType.X) or (v0t == VertexType.X and v1t == VertexType.Z))):
                v0n = [n for n in g.getNeighborsConcrete(v0) if not n == v1]
                v1n = [n for n in g.getNeighborsConcrete(v1) if not n == v0]
                if (
                    all([types(n) == v1t and phases(n) == 0 for n in v0n]) and
                    all([types(n) == v0t and phases(n) == 0 for n in v1n])):
                    i += 1
                    for v in v0n:
                        for c in g.incident_edges(v): candidates.discard(c)
                    for v in v1n:
                        for c in g.incident_edges(v): candidates.discard(c)
                    m.append((v0,v1,v0n,v1n))
    return m

def bialg(g: EGraph, matches: List[MatchBialgType[VT]]) -> RewriteOutputType[ET,VT]:
    """Performs a certain type of bialgebra rewrite given matchings supplied by
    ``match_bialg(_parallel)``."""
    rem_verts = []
    etab: Dict[ET, List[int]] = dict()
    for m in matches:
        rem_verts.append(m[0])
        rem_verts.append(m[1])
        es = [(i,j) for i in m[2] for j in m[3]]
        for e in es:
            if e in etab: etab[e][0] += 1
            else: etab[e] = [1,0]
    
    return (etab, rem_verts, [], True)

def simp(
    g: EGraph,
    name: str,
    match: Callable[..., List[MatchObject]],
    rewrite: Callable[[EGraph,List[MatchObject]],RewriteOutputType[ET,VT]],
    matchf:Optional[Union[Callable[[ET],bool], Callable[[VT],bool]]]=None,
    quiet:bool=False,
    stats:Optional[Stats]=None,
    maxiter:Optional[int]=None) -> int:
    """Helper method for constructing simplification strategies based on the rules present in rules_.
    It uses the ``match`` function to find matches, and then rewrites ``g`` using ``rewrite``.
    If ``matchf`` is supplied, only the vertices or edges for which matchf() returns True are considered for matches.

    Example:
        ``simp(g, 'spider_simp', rules.match_spider_parallel, rules.spider)``

    Args:
        g: The graph that needs to be simplified.
        str name: The name to display if ``quiet`` is set to False.
        match: One of the ``match_*`` functions of rules_.
        rewrite: One of the rewrite functions of rules_.
        matchf: An optional filtering function on candidate vertices or edges, which
           is passed as the second argument to the match function.
        quiet: Suppress output on numbers of matches found during simplification.
        maxiter: The maximum number of iterations to try.

    Returns:
        Number of iterations of ``rewrite`` that had to be applied before no more matches were found."""

    i = 0
    new_matches = True
    while new_matches:
        if maxiter and i >= maxiter:
            break
        new_matches = False
        if matchf is not None:
            m = match(g, matchf)
        else:
            m = match(g)
        if len(m) > 0:
            i += 1
            if i == 1 and not quiet: print("{}: ".format(name),end='')
            if not quiet: print(len(m), end='')
            #print(len(m), end='', flush=True) #flush only supported on Python >3.3
            etab, rem_verts, rem_edges, check_isolated_vertices = rewrite(g, m)
            g.add_edge_table(etab)
            g.remove_edges(rem_edges)
            g.remove_vertices(rem_verts)
            if check_isolated_vertices: g.remove_isolated_vertices()
            if not quiet: print('. ', end='')
            new_matches = True
            if stats is not None: stats.count_rewrites(name, len(m))
    if not quiet and i>0: print(' {!s} iterations'.format(i))
    return i

def bialg_simp(g: EGraph, quiet:bool=False, stats: Optional[Stats]=None) -> int:
    return simp(g, 'bialg_simp', match_bialg_parallel, bialg, quiet=quiet, stats=stats)
