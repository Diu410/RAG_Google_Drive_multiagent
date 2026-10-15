"""Query-directed traversal of cached source references, without model calls.

Edges mean a shared literal name/identifier, or a semantic document link. They
are retrieval hints, never verified relations. Only fully covered, explicitly
anchored questions can bypass ordinary hybrid retrieval.
"""
from collections import OrderedDict, defaultdict
import re

from index_transactions import consistent_read, index_revision
from source_grounding import factual_source_text
from text_normalization import lexical_key

_WORDS = re.compile(r'[^\W_]+(?:[-./][^\W_]+)*', re.UNICODE)
_REFERENCES = re.compile(r'(?<!\w)(?:[A-ZА-ЯЁ][\w.-]{2,}|[\w]+[-./][\w]*\d[\w.-]*)(?!\w)')
_BROAD = re.compile(
    r'\b(?:сравн\w*|сопостав\w*|анализ\w*|проанализ\w*|обзор\w*|подроб\w*|'
    r'расскажи\w*|опиши\w*|все|всех|полн\w*|почему|зачем|compare|summari[sz]\w*|'
    r'describe|explain|overview|comprehensive|why|all)\b|'
    r'\b(?:кто\s+так\w*|что\s+(?:такое|за|знаешь)|who\s+is)\b', re.I)
_STOP_WORDS = set('''а и в во на по с со к ко от до из у о об для при после перед
    как кто что где какой какая какое какие какого каком когда сколько ли это
    этого этой этот этих его ее их он она они мы вы ты мне пожалуйста нужно
    необходимо обязательно должен должна должны выполнять выполняется
    the a an of in on at to for from by with and or is are was were be been
    what which who where when how long must should does do did please include
    including after before it its this that their каковы указаны указан назови'''.split())
_cache = OrderedDict()
MAX_VISITED = 128
MAX_HOPS = 2
_QUALIFIED = re.compile(r'\b(?:не|нет|not|never|except|кроме|без|раньше|позже|последн\w*|latest|earliest)\b', re.I)
_FORMULA = re.compile(r'[\u0370-\u03ff∑∫√±×÷≤≥≠≈∞∂∇=+*/^%₀-₉]')


def term_key(word):
    key = lexical_key(word.strip('.-/'))
    return key[:5] if key.isalpha() and len(key) >= 6 else key


def reference_key(word):
    # Names/identifiers must not collide merely because their prefixes match.
    return lexical_key(word.strip('.-/'))


_STOP = {term_key(w) for w in _STOP_WORDS}
_ALIASES = {'изгот': {'изгот', 'произ', 'выпус'},
            'произ': {'изгот', 'произ', 'выпус'},
            'manuf': {'manuf', 'makes', 'produ'},
            'наход': {'наход', 'распо'}, 'locat': {'locat', 'based'}}


def query_terms(text):
    words = _WORDS.findall(text)
    terms = {term_key(w) for w in words
             if (len(w) >= 3 or any(c.isdigit() for c in w) or len(w) >= 2 and w[0].isupper())
             and term_key(w) not in _STOP}
    return terms | literal_references(text)


def covered_terms(wanted, present):
    return {term for term in wanted if term in present or _ALIASES.get(term, set()) & present}


def literal_references(text):
    references = set()
    for match in _REFERENCES.finditer(text):
        # Sentence-initial ordinary words are not automatically named objects.
        prefix = text[:match.start()].rstrip()
        identifier = bool(re.search(r'\d', match[0]))
        if not identifier and (not prefix or prefix[-1:] in '.!?\n'):
            continue
        key = reference_key(match[0])
        if len(key) >= 4 and key not in _STOP:
            references.add(key)
    return references


class GraphQueryIndex:
    def __init__(self, graph, snapshot, corpus):
        self.graph = graph
        self.snapshot = snapshot
        self.rows, self.terms, self.references = {}, {}, {}
        self.by_reference, self.by_document = defaultdict(set), defaultdict(set)
        self.adjacent = defaultdict(set)
        inventory = snapshot['inventory']
        for row in corpus.records:
            owner = graph.document_key(row.metadata, inventory)
            if owner is None or row.chunk_id not in snapshot['chunks']:
                continue
            text = factual_source_text(row.text)
            self.rows[row.chunk_id] = row
            self.terms[row.chunk_id] = query_terms(text)
            self.references[row.chunk_id] = literal_references(text)
            self.by_document[owner].add(row.chunk_id)
            for ref in self.references[row.chunk_id]:
                self.by_reference[ref].add(row.chunk_id)
        for left, right in snapshot['edges']:
            self.adjacent[left].add(right)
            self.adjacent[right].add(left)

    def neighbors(self, chunk_id, *, semantic=True):
        owner = self.snapshot['chunks'][chunk_id][0]
        result = set(self.by_document[owner])
        if semantic:
            for other in self.adjacent[owner]:
                result.update(self.by_document[other])
        for ref in self.references[chunk_id]:
            # Very common names form hubs, not a bounded useful path.
            posting = self.by_reference[ref]
            if len(posting) <= 32:
                result.update(posting)
        return result - {chunk_id}

    def search(self, question, limit):
        if (_BROAD.search(question) or _QUALIFIED.search(question) or _FORMULA.search(question)
                or self.snapshot['truncated'] or limit < 1):
            return None
        wanted = query_terms(question)
        anchors = {reference_key(w) for w in _WORDS.findall(question)} & self.by_reference.keys()
        if not wanted or not anchors:
            return None
        anchor_terms = anchors | {term_key(a) for a in anchors}
        if not wanted - anchor_terms - {'проек', 'proje', 'устро', 'devic', 'предп', 'compa'}:
            return None
        roots = set.union(*(self.by_reference[key] for key in anchors))
        if not roots or len(roots) > 32:
            return None
        # All explicit objects must be covered. A different year is never an
        # interchangeable root, but a bridge need not repeat the root's year.
        years = {term for term in wanted if re.fullmatch(r'(?:19|20)\d{2}', term)}
        roots = {cid for cid in roots if anchors <= self.references[cid] and (
            not years or not (set(re.findall(r'\b(?:19|20)\d{2}\b', self.rows[cid].text)) - years))}
        if not roots:
            return None
        paths = {cid: (cid,) for cid in sorted(roots)}
        frontier = sorted(roots)
        for _ in range(MAX_HOPS):
            following = []
            for cid in frontier:
                for other in sorted(self.neighbors(cid, semantic=False)):
                    if years and set(re.findall(r'\b(?:19|20)\d{2}\b', self.rows[other].text)) - years:
                        continue
                    if other not in paths:
                        paths[other] = paths[cid] + (other,)
                        following.append(other)
                        if len(paths) > MAX_VISITED:
                            return None
            frontier = following
        coverage = {cid: covered_terms(wanted, self.terms[cid]) for cid in paths}
        if set.union(*coverage.values()) != wanted:
            return None
        selected, found = [], set()
        while found != wanted:
            options = []
            for cid, path in paths.items():
                missing_ids = [part for part in path if part not in selected]
                if not missing_ids or len(selected) + len(missing_ids) > limit:
                    continue
                gain = set.union(*(coverage[part] for part in path)) - found
                if gain:
                    options.append((len(gain) / len(missing_ids), len(gain), -len(missing_ids), cid))
            if not options:
                return None
            cid = max(options)[-1]
            for part in paths[cid]:
                if part not in selected:
                    selected.append(part)
                    found.update(coverage[part])
        # Retain alternative passages on the same explicit object and requested
        # detail. Otherwise selecting one conflicting statement hides the other.
        details = wanted - anchor_terms
        alternatives = [cid for cid in sorted(roots) if cid not in selected and (
            not details or len(coverage[cid] & details) >= max(1, len(details) * .75))]
        if len(selected) + len(alternatives) > limit:
            return None
        selected.extend(alternatives)
        return self.read(selected, paths, fast=True)

    def expand(self, question, seeds, limit=1):
        if _BROAD.search(question) or _QUALIFIED.search(question) or _FORMULA.search(question):
            return []
        wanted = query_terms(question)
        found = set.union(*(covered_terms(wanted, query_terms(s.text)) for s in seeds)) if seeds else set()
        missing = wanted - found
        if not missing or not seeds or limit < 1:
            return []
        def seed_owner(seed):
            if self.graph.workspace_id and seed.metadata.get('workspace_id') == self.graph.workspace_id:
                path = seed.metadata.get('relative_path')
                return path if path in self.by_document else None
            return self.graph.document_key(seed.metadata, self.snapshot['inventory'])
        owners = {seed_owner(s) for s in seeds}
        roots = []
        for seed in seeds[:3]:
            owner = seed_owner(seed)
            if owner is None or not self.graph._file_is_current(owner, self.snapshot['inventory']):
                continue
            if seed.chunk_id in self.rows and self.snapshot['chunks'][seed.chunk_id][0] == owner:
                roots.append(seed.chunk_id)
                continue
            ids = self.by_document.get(owner, set())
            ranked = sorted(ids, key=lambda cid: (-len(covered_terms(wanted, self.terms[cid])), cid))
            if ranked and covered_terms(wanted, self.terms[ranked[0]]):
                roots.append(ranked[0])
        paths = {cid: (cid,) for cid in roots}
        frontier = roots
        years = {term for term in wanted if re.fullmatch(r'(?:19|20)\d{2}', term)}
        for _ in range(MAX_HOPS):
            following = []
            for cid in frontier:
                for other in sorted(self.neighbors(cid)):
                    if other not in paths:
                        if years and set(re.findall(r'\b(?:19|20)\d{2}\b', self.rows[other].text)) - years:
                            continue
                        paths[other] = paths[cid] + (other,)
                        following.append(other)
                        if len(paths) > MAX_VISITED:
                            return []
            frontier = following
        options = []
        for cid, path in paths.items():
            if self.snapshot['chunks'][cid][0] in owners:
                continue
            new_ids = [part for part in path if self.snapshot['chunks'][part][0] not in owners]
            gain = set.union(*(covered_terms(missing, self.terms[part]) for part in new_ids)) if new_ids else set()
            if gain and len(new_ids) <= limit:
                options.append((len(gain), -len(new_ids), cid, new_ids))
        if not options:
            return []
        _, _, _, ids = max(options)
        return self.read(ids, paths) or []

    def read(self, ids, paths, *, fast=False):
        from hybrid_context_extractor import SearchResult
        if not ids:
            return []
        path_ids = list(dict.fromkeys(part for cid in ids for part in paths[cid]))
        stored = self.graph.store.get(ids=path_ids, include=['documents', 'metadatas'])
        fresh = {cid: (text, meta or {}) for cid, text, meta in zip(
            stored.get('ids') or [], stored.get('documents') or [], stored.get('metadatas') or [])}
        checked_owners = set()
        for cid in path_ids:
            text, meta = fresh.get(cid, ('', {}))
            owner = self.graph.document_key(meta, self.snapshot['inventory'])
            original_owner = self.snapshot['chunks'][cid][0]
            if (not text or owner != original_owner or text != self.rows[cid].text
                    or (owner not in checked_owners and not self.graph._file_is_current(owner, self.snapshot['inventory']))):
                return None
            checked_owners.add(owner)
        output = []
        for cid in ids:
            text, meta = fresh[cid]
            owner = self.snapshot['chunks'][cid][0]
            path = paths[cid]
            label = self.snapshot['inventory'][owner]['label']
            document_id = f'workspace:{self.graph.workspace_id}:{owner}' if self.graph.workspace_id else int(owner)
            metadata = {**meta, 'document_id': document_id, 'chunk_id': cid, 'source': label,
                        'graph_path': list(path), 'graph_fast_path': fast,
                        'graph_anchor': self.snapshot['inventory'][self.snapshot['chunks'][path[0]][0]]['label']}
            output.append(SearchResult(document_id, cid, text, metadata, 1.0, ['semantic_graph']))
        return output


@consistent_read
def query_index(graph):
    from hybrid_context_extractor import cached_corpus
    snapshot = graph.snapshot()
    corpus = cached_corpus(graph.store)
    key = (id(graph.store), graph.workspace_id, graph.connection_factory)
    signature = (id(snapshot), id(corpus), index_revision())
    cached = _cache.get(key)
    if cached is None or cached[0] != signature or cached[1].graph.store is not graph.store:
        cached = (signature, GraphQueryIndex(graph, snapshot, corpus))
        _cache[key] = cached
    _cache.move_to_end(key)
    while len(_cache) > 4:
        _cache.popitem(last=False)
    return cached[1]


def graph_search(store, question, limit=3, *, workspace_id=None,
                 connection_factory=None, filesystem=None):
    from semantic_graph import SemanticDocumentGraph
    kwargs = {'filesystem': filesystem}
    if connection_factory is not None:
        kwargs['connection_factory'] = connection_factory
    if store is None or _BROAD.search(question) or _QUALIFIED.search(question) or _FORMULA.search(question):
        return None
    try:
        graph = SemanticDocumentGraph(store, workspace_id, **kwargs)
        if graph.snapshot()['truncated']:
            return None
        return query_index(graph).search(question, limit)
    except Exception:
        import logging
        logging.getLogger(__name__).debug('Graph retrieval unavailable; using hybrid', exc_info=True)
        return None
