"""Relational features via javalang (no compilation, no type resolution).

Class level: ATFD (God Class rule), FDP, and the Data Class rule inputs NOPA, NOAM, WOC.
Method level (D-6, stored for the parked method level): ATFD, LAA, FDP per method.

Definitions (Lanza & Marinescu, approximated syntactically):
  foreign attribute access = reading/writing `recv.field`, or calling an accessor
      `recv.getX()` / `recv.isX()` / `recv.setX(v)`, where `recv` is not this/super and its
      declared type is not the class itself, its superclass, or an ignored JDK type.
      Accessors map to attribute names (getFoo -> foo) so `o.foo` and `o.getFoo()` count once.
  own attribute access = bare or `this.`-qualified reference to a field declared in the class
      (or an enclosing class), or an unqualified accessor call.
  ATFD = distinct (receiver type, attribute) foreign accesses.
  FDP  = distinct receiver types among those foreign accesses.
  LAA  = own / (own + foreign) distinct accesses; NaN when a method accesses no attributes.
  NOPA = public non-static attributes declared in the class.
  NOAM = public non-static accessor methods (getX/isX with 0 args, setX with 1 arg), by name.
  WOC  = public functional (non-accessor) methods / (public methods + NOPA); constructors excluded.
         NaN when the class has no public members. Interface methods count as public.

Known approximations (record them in the writeup):
  - Receiver types come from declared types of locals, parameters and fields. Receivers that
    cannot be resolved (method-call chains, unknown names) are keyed by expression text, so
    each counts as its own provider in FDP.
  - ALL_CAPS constants are not counted: neither bare own constants nor `Type.CONSTANT`.
  - Anonymous/local class bodies inside a method count toward that method.
  - javalang 0.13 cannot parse two Java 8 constructs. When a file fails to parse, they are
    blanked out with spaces (same lines and columns) and the parse retried, one rewrite at a
    time, cumulatively:
      1. array constructor references `Type[]::new` -> `Type  ::new`; method references are
         not attribute accesses here;
      2. type-use annotations in type arguments `Map<K, @Nullable V>` -> `Map<K,           V>`;
         annotation uses never feed a metric.
    So no metric changes.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import javalang
from javalang import tree as T

ACCESSOR_RE = re.compile(r"^(get|is|set)([A-Z]\w*)$")
CONSTANT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
ARRAY_CTOR_REF_RE = re.compile(r"([\w.]+\s*)((?:\[\s*\]\s*)+)(::\s*new\b)")
TYPE_ARG_ANNOTATION_RE = re.compile(r"([<,]\s*)(@[A-Za-z_][\w.]*)(?=\s+[A-Za-z_?])")
PARSE_ERRORS = (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, TypeError,
                AttributeError, IndexError, RecursionError, StopIteration)

CLASS_COLUMNS = ["file", "class_fqn", "simple_name", "kind", "line", "n_methods",
                 "atfd", "fdp", "n_own_attrs", "nopa", "noam", "n_public_methods", "woc",
                 "parse_error"]
METHOD_COLUMNS = ["file", "class_fqn", "method", "params", "line", "is_constructor",
                  "atfd", "laa", "fdp", "n_own_attrs", "n_foreign_attrs"]


def _type_name(t) -> str | None:
    """Simple name of a declared type; None for primitives and arrays (no attributes)."""
    if t is None or isinstance(t, T.BasicType):
        return None
    if getattr(t, "dimensions", None) and any(d is not None for d in t.dimensions):
        return None
    name = t.name
    sub = getattr(t, "sub_type", None)
    while sub is not None:
        name = sub.name
        sub = getattr(sub, "sub_type", None)
    return name


def _type_str(t) -> str:
    if t is None:
        return "?"
    if isinstance(t, T.BasicType):
        base = t.name
    else:
        parts, cur = [], t
        while cur is not None:
            s = cur.name
            if getattr(cur, "arguments", None):
                s += "<" + ",".join(_type_str(a.type) if a.type is not None else "?" for a in cur.arguments) + ">"
            parts.append(s)
            cur = getattr(cur, "sub_type", None)
        base = ".".join(parts)
    dims = getattr(t, "dimensions", None) or []
    return base + "[]" * len(dims)


def _accessor_attr(name: str, n_args: int) -> str | None:
    m = ACCESSOR_RE.match(name)
    if not m:
        return None
    prefix, rest = m.groups()
    if (prefix in ("get", "is") and n_args != 0) or (prefix == "set" and n_args != 1):
        return None
    return rest[0].lower() + rest[1:]


def _field_types(body: Iterable) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for member in body or []:
        if isinstance(member, T.FieldDeclaration):
            for d in member.declarators:
                out[d.name] = _type_name(member.type)
    return out


def _local_types(method) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for prm in method.parameters or []:
        out[prm.name] = _type_name(prm.type)
    for _, node in method.filter(T.LocalVariableDeclaration):
        for d in node.declarators:
            out[d.name] = _type_name(node.type)
    for _, node in method.filter(T.VariableDeclaration):
        for d in node.declarators:
            out.setdefault(d.name, _type_name(node.type))
    for _, node in method.filter(T.FormalParameter):
        out.setdefault(node.name, _type_name(node.type))
    for _, node in method.filter(T.CatchClauseParameter):
        out.setdefault(node.name, None)
    for _, node in method.filter(T.LambdaExpression):
        for prm in node.parameters or []:
            name = getattr(prm, "name", None) or getattr(prm, "member", None)
            if name:
                out.setdefault(name, _type_name(getattr(prm, "type", None)))
    return out


def _analyze_method(method, own_fields: dict, own_types: set[str], ignored: set[str]):
    locals_ = _local_types(method)
    own: set[str] = set()
    foreign: set[tuple[str, str]] = set()

    # Selectors: `this.x` / `this.getX()` are own accesses; chained selectors
    # (`a.getB().getC()`) have an unresolvable receiver keyed by the preceding member.
    this_qualified: set[int] = set()
    chained: dict[int, str] = {}
    for _, node in method:
        selectors = getattr(node, "selectors", None)
        if not selectors:
            continue
        prev = getattr(node, "member", None) or type(node).__name__
        for i, sel in enumerate(selectors):
            if isinstance(node, T.This) and i == 0:
                this_qualified.add(id(sel))
            else:
                chained[id(sel)] = f"?{prev}()"
            prev = getattr(sel, "member", prev)

    def resolve(qualifier: str) -> tuple[str, bool]:
        """Return (receiver key, is_own)."""
        head = qualifier.split(".")[0]
        if head in ("this", "super"):
            return qualifier, True
        if qualifier in locals_:
            t = locals_[qualifier]
        elif qualifier in own_fields:
            t = own_fields[qualifier]
        elif head[:1].isupper():
            t = qualifier.split(".")[-1]
        else:
            t = None
        if t is not None and t in own_types:
            return t, True
        return (t or f"?{qualifier}"), False

    for _, node in method:
        if isinstance(node, T.MemberReference):
            member, q = node.member, node.qualifier or ""
            if id(node) in this_qualified:
                if member in own_fields:
                    own.add(member)
                continue
            if id(node) in chained:
                foreign.add((chained[id(node)], member))
                continue
            if not q:
                if member in own_fields and member not in locals_ and not CONSTANT_RE.match(member):
                    own.add(member)
                continue
            if CONSTANT_RE.match(member) and q.split(".")[-1][:1].isupper():
                continue
            rtype, is_own = resolve(q)
            if is_own:
                own.add(member)
            elif rtype not in ignored:
                foreign.add((rtype, member))
        elif isinstance(node, T.MethodInvocation):
            attr = _accessor_attr(node.member, len(node.arguments or []))
            if attr is None:
                continue
            q = node.qualifier or ""
            if id(node) in this_qualified or (not q and id(node) not in chained):
                own.add(attr)
                continue
            if id(node) in chained:
                foreign.add((chained[id(node)], attr))
                continue
            rtype, is_own = resolve(q)
            if is_own:
                own.add(attr)
            elif rtype not in ignored:
                foreign.add((rtype, attr))
        elif isinstance(node, T.SuperMethodInvocation):
            attr = _accessor_attr(node.member, len(node.arguments or []))
            if attr:
                own.add(attr)
    return own, foreign


def _members(type_node) -> list:
    body = type_node.body
    if isinstance(body, T.EnumBody):
        return list(body.declarations or [])
    return list(body or [])


def _public_member_counts(members: list, is_interface: bool) -> dict:
    nopa = sum(len(m.declarators) for m in members
               if isinstance(m, T.FieldDeclaration)
               and "public" in m.modifiers and "static" not in m.modifiers)
    n_public = noam = 0
    for m in members:
        if not isinstance(m, T.MethodDeclaration) or "static" in m.modifiers:
            continue
        if "public" in m.modifiers or (is_interface and "private" not in m.modifiers):
            n_public += 1
            if _accessor_attr(m.name, len(m.parameters or [])):
                noam += 1
    denom = n_public + nopa
    return {"nopa": nopa, "noam": noam, "n_public_methods": n_public,
            "woc": (n_public - noam) / denom if denom else math.nan}


def _line(node) -> int | None:
    pos = getattr(node, "position", None)
    return pos.line if pos else None


def _walk_types(type_node, prefix: str, outer_fields: dict, ignored: set[str],
                file: str, class_rows: list, method_rows: list, method_level: bool) -> None:
    if isinstance(type_node, T.AnnotationDeclaration):
        return
    fqn = f"{prefix}.{type_node.name}" if prefix else type_node.name
    members = _members(type_node)
    own_fields = {**outer_fields, **_field_types(members)}
    own_types = {type_node.name}
    ext = getattr(type_node, "extends", None)
    if isinstance(ext, list):  # interfaces may extend several
        own_types.update(n for n in (_type_name(e) for e in ext) if n)
    elif ext is not None:
        n = _type_name(ext)
        if n:
            own_types.add(n)

    class_own: set[str] = set()
    class_foreign: set[tuple[str, str]] = set()
    n_methods = 0
    for m in members:
        if isinstance(m, (T.MethodDeclaration, T.ConstructorDeclaration)):
            n_methods += 1
            own, foreign = _analyze_method(m, own_fields, own_types, ignored)
            class_own |= own
            class_foreign |= foreign
            if method_level:
                total = len(own) + len(foreign)
                method_rows.append({
                    "file": file, "class_fqn": fqn, "method": m.name,
                    "params": ",".join(_type_str(prm.type) + ("..." if getattr(prm, "varargs", False) else "")
                                       for prm in m.parameters or []),
                    "line": _line(m), "is_constructor": isinstance(m, T.ConstructorDeclaration),
                    "atfd": len(foreign), "laa": (len(own) / total) if total else math.nan,
                    "fdp": len({t for t, _ in foreign}),
                    "n_own_attrs": len(own), "n_foreign_attrs": len(foreign),
                })
    kind = {T.ClassDeclaration: "class", T.InterfaceDeclaration: "interface",
            T.EnumDeclaration: "enum"}.get(type(type_node), type(type_node).__name__)
    class_rows.append({
        "file": file, "class_fqn": fqn, "simple_name": type_node.name, "kind": kind,
        "line": _line(type_node), "n_methods": n_methods, "atfd": len(class_foreign),
        "fdp": len({t for t, _ in class_foreign}), "n_own_attrs": len(class_own),
        **_public_member_counts(members, isinstance(type_node, T.InterfaceDeclaration)),
        "parse_error": None,
    })
    for m in members:
        if isinstance(m, (T.ClassDeclaration, T.InterfaceDeclaration, T.EnumDeclaration)):
            _walk_types(m, fqn, own_fields, ignored, file, class_rows, method_rows, method_level)


def blank_array_ctor_refs(source: str) -> str:
    """`Type[]::new` -> `Type  ::new`, keeping every line and column where it was."""
    return ARRAY_CTOR_REF_RE.sub(lambda m: m.group(1) + " " * len(m.group(2)) + m.group(3), source)


def blank_type_arg_annotations(source: str) -> str:
    """`Map<K, @Nullable V>` -> `Map<K,           V>`, keeping every line and column where it was."""
    return TYPE_ARG_ANNOTATION_RE.sub(lambda m: m.group(1) + " " * len(m.group(2)), source)


def parse_java(source: str):
    """javalang parse; on a syntax error, retried with the rewrites in the docstring applied
    cumulatively. Re-raises the original error if no rewrite applies or none helps."""
    try:
        return javalang.parse.parse(source)
    except javalang.parser.JavaSyntaxError as first:
        current = source
        for rewrite in (blank_array_ctor_refs, blank_type_arg_annotations):
            retry = rewrite(current)
            if retry == current:
                continue
            current = retry
            try:
                return javalang.parse.parse(current)
            except javalang.parser.JavaSyntaxError:
                pass
        raise first


def analyze_source(source: str, file: str, ignored_types: Iterable[str] = (),
                   method_level: bool = True) -> tuple[list[dict], list[dict]]:
    """Parse one Java file. On a parse error returns a single class row carrying the error."""
    class_rows: list[dict] = []
    method_rows: list[dict] = []
    try:
        cu = parse_java(source)
    except PARSE_ERRORS as e:
        err = f"{type(e).__name__}: {getattr(e, 'description', '') or str(e)}"[:300]
        return [{**{c: None for c in CLASS_COLUMNS}, "file": file, "parse_error": err}], []
    pkg = cu.package.name if cu.package else ""
    ignored = set(ignored_types)
    for t in cu.types or []:
        _walk_types(t, pkg, {}, ignored, file, class_rows, method_rows, method_level)
    return class_rows, method_rows
