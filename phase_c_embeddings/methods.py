"""Step 9 — split a class snippet into the methods hierarchical pooling encodes (D-12, PRD §9).

Members: every method and constructor in the snippet — those of the sample's type, of its nested
types and of enum-constant bodies — in source order, each tagged with `nested_depth` (0 = declared in
the sample's own type). Anonymous and local classes stay part of the method that contains them.
Initializer blocks, fields and annotation-type elements are not methods.

Text: from the method's leading Javadoc (`/** ... */` directly before it; `method_text: with_javadoc`)
or else its first annotation / modifier, to its last token (closing brace, or `;` for abstract and
interface methods), sliced from the original snippet. javalang has no end positions and a member's
`position` starts after its annotations and modifiers, so the boundaries come from the token indices
the parser consumed for each member (`_SpanParser`), not from `position`.

Parsing applies the same blank-out rewrites as relational.parse_java (`Type[]::new`, type-use
annotations in type arguments); they keep every line and column, so offsets into the rewritten text
are valid for the original. Each slice is checked against the token values it should start and end
with; a mismatch is reported as an error for the whole class (hierarchical pooling then falls back
to truncation for it).
"""
from __future__ import annotations

from dataclasses import dataclass

import javalang
from javalang import tree as T

from phase_a_data.relational import PARSE_ERRORS, _members, blank_array_ctor_refs, blank_type_arg_annotations

EXTRACTOR_VERSION = "1"   # part of the Step 9 fingerprint: bump on any change to what is extracted
TYPES = (T.ClassDeclaration, T.InterfaceDeclaration, T.EnumDeclaration)
METHODS = (T.MethodDeclaration, T.ConstructorDeclaration)


@dataclass
class Method:
    ordinal: int            # source order within the snippet, from 0
    name: str
    kind: str               # method | constructor | abstract (no body)
    nested_depth: int
    owner: str              # dotted path of the declaring type inside the snippet, e.g. Outer.Inner
    start_line: int         # 1-based, within the snippet
    end_line: int
    has_javadoc: bool
    empty_body: bool
    text: str


class _SpanParser(javalang.parser.Parser):
    """Records, for every class/interface body member, the first and last token index it consumed."""

    def __init__(self, tokens):
        super().__init__(tokens)
        self.spans: dict[int, tuple[int, int]] = {}

    def _record(self, parse):
        start = self.tokens.marker
        node = parse()
        if node is not None:
            self.spans[id(node)] = (start, self.tokens.marker - 1)
        return node

    def parse_class_body_declaration(self):
        return self._record(super().parse_class_body_declaration)

    def parse_interface_body_declaration(self):
        return self._record(super().parse_interface_body_declaration)


def _parse(source: str):
    """(compilation unit, tokens, spans, text actually parsed). Same retry rewrites as parse_java."""
    def attempt(text):
        tokens = list(javalang.tokenizer.tokenize(text))
        parser = _SpanParser(tokens)
        return parser.parse(), tokens, parser.spans, text

    try:
        return attempt(source)
    except javalang.parser.JavaSyntaxError as first:
        current = source
        for rewrite in (blank_array_ctor_refs, blank_type_arg_annotations):
            retry = rewrite(current)
            if retry == current:
                continue
            current = retry
            try:
                return attempt(current)
            except javalang.parser.JavaSyntaxError:
                pass
        raise first


def _javadoc_start(source: str, prev_end: int, start: int) -> int | None:
    """Offset of a `/** ... */` comment that ends right before `start` (only whitespace between),
    searched no further back than the previous token's end."""
    gap = source[prev_end:start]
    body = gap.rstrip()
    if not body.endswith("*/"):
        return None
    j = body.rfind("/**")
    if j < 0 or "*/" in body[j + 3:-2]:
        return None
    return prev_end + j


def _walk(type_node, owner: str, depth: int, out: list):
    for m in _members(type_node):
        if isinstance(m, METHODS):
            out.append((m, owner, depth))
        elif isinstance(m, TYPES):
            _walk(m, f"{owner}.{m.name}", depth + 1, out)
    if isinstance(type_node, T.EnumDeclaration):
        for c in type_node.body.constants or []:
            for m in c.body or []:
                if isinstance(m, METHODS):
                    out.append((m, f"{owner}.{c.name}", depth + 1))
                elif isinstance(m, TYPES):
                    _walk(m, f"{owner}.{c.name}.{m.name}", depth + 2, out)


def extract_methods(source: str, with_javadoc: bool = True) -> list[Method]:
    """Methods of a class snippet in source order. Raises a PARSE_ERRORS exception if the snippet does
    not parse, and ValueError if a computed boundary does not match its tokens."""
    cu, tokens, spans, parsed = _parse(source)
    line_start = [0] + [i + 1 for i, c in enumerate(source) if c == "\n"]

    def offset(tok) -> int:
        return line_start[tok.position.line - 1] + tok.position.column - 1

    found: list = []
    for t in cu.types or []:
        if isinstance(t, TYPES):
            _walk(t, t.name, 0, found)
    if any(id(m) not in spans for m, _o, _d in found):
        bad = next(f"{o}.{m.name}" for m, o, _d in found if id(m) not in spans)
        raise ValueError(f"no token span recorded for {bad}")
    found.sort(key=lambda f: spans[id(f[0])][0])        # source order (enum-constant bodies interleave)
    out = []
    for m, owner, depth in found:
        i0, i1 = spans[id(m)]
        first, last = tokens[i0], tokens[i1]
        start, end = offset(first), offset(last) + len(last.value)
        if parsed[start:start + len(first.value)] != first.value or parsed[end - len(last.value):end] != last.value:
            raise ValueError(f"boundary mismatch for {owner}.{m.name} at line {first.position.line}")
        doc = None
        if with_javadoc:
            prev_end = offset(tokens[i0 - 1]) + len(tokens[i0 - 1].value) if i0 > 0 else 0
            doc = _javadoc_start(source, prev_end, start)
            if doc is not None:
                start = doc
        body = getattr(m, "body", None)
        kind = "constructor" if isinstance(m, T.ConstructorDeclaration) else ("abstract" if body is None else "method")
        out.append(Method(
            ordinal=len(out), name=m.name, kind=kind, nested_depth=depth, owner=owner,
            start_line=source.count("\n", 0, start) + 1, end_line=source.count("\n", 0, end - 1) + 1,
            has_javadoc=doc is not None, empty_body=body is not None and len(body) == 0,
            text=source[start:end]))
    return out


__all__ = ["EXTRACTOR_VERSION", "Method", "extract_methods", "PARSE_ERRORS"]
