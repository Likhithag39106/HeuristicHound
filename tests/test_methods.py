"""Step 9 method extraction (D-12): exact boundaries from token indices, nested types, Javadoc."""
import pytest

from phase_c_embeddings.methods import PARSE_ERRORS, extract_methods

SRC = '''public class A<T> extends B {
    private int x;
    /** Doc for f. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    @Override
    public <U> List<U> f(int a) throws IOException {
        Runnable r = () -> { };
        Object o = new Object() { public String toString() { return "}"; } };
        return null;
    }
    // line comment, not Javadoc
    A() {}
    /* block comment, not Javadoc */
    abstract void g();
    static class Inner { void h() { int[] q = {1, 2}; } }
    enum E { X { void k() {} }; void m() {} }
    IntFunction<String[]> s = String[]::new;
    static { x(); }
}'''


def _by_name(src=SRC, **kw):
    return {m.name: m for m in extract_methods(src, **kw)}


def test_annotations_javadoc_and_braces_inside_are_kept_whole():
    f = _by_name()["f"]
    assert f.text.startswith("/** Doc for f. */\n    @SuppressWarnings({")
    assert f.text.endswith("return null;\n    }")
    assert "toString()" in f.text                     # anonymous class stays inside its method
    assert (f.start_line, f.end_line, f.has_javadoc, f.kind) == (3, 10, True, "method")


def test_code_only_starts_at_first_annotation():
    f = _by_name(with_javadoc=False)["f"]
    assert f.text.startswith('@SuppressWarnings({"unchecked"') and not f.has_javadoc
    assert f.start_line == 4


def test_comments_that_are_not_javadoc_are_left_out():
    ms = _by_name()
    assert ms["A"].text == "A() {}" and not ms["A"].has_javadoc
    assert ms["g"].text == "abstract void g();" and ms["g"].kind == "abstract"


def test_kinds_empty_bodies_nesting_and_source_order():
    ms = extract_methods(SRC)
    assert [m.name for m in ms] == ["f", "A", "g", "h", "k", "m"]
    assert [m.ordinal for m in ms] == list(range(6))
    by = {m.name: m for m in ms}
    assert by["A"].kind == "constructor" and by["A"].empty_body
    assert (by["h"].owner, by["h"].nested_depth) == ("A.Inner", 1)
    assert (by["k"].owner, by["k"].nested_depth) == ("A.E.X", 2)   # enum-constant body
    assert (by["m"].owner, by["m"].nested_depth) == ("A.E", 1)
    assert by["h"].text == "void h() { int[] q = {1, 2}; }"


def test_slices_come_from_the_original_text_after_a_parse_rewrite():
    src = "class C {\n    Object[] f(List<String> l) { return l.stream().toArray(String[]::new); }\n}\n"
    (m,) = extract_methods(src)
    assert m.text == "Object[] f(List<String> l) { return l.stream().toArray(String[]::new); }"


def test_crlf_tabs_and_non_ascii_offsets():
    src = "class C {\r\n\t/** déjà */\r\n\tint f() {\r\n\t\treturn \"中\";\r\n\t}\r\n}\r\n"
    (m,) = extract_methods(src)
    assert m.text == "/** déjà */\r\n\tint f() {\r\n\t\treturn \"中\";\r\n\t}"


def test_interfaces_and_no_methods():
    ms = extract_methods("public interface I {\n  @Deprecated\n  void a();\n  default int b() { return 1; }\n}")
    assert [(m.name, m.kind, m.text) for m in ms] == [
        ("a", "abstract", "@Deprecated\n  void a();"), ("b", "method", "default int b() { return 1; }")]
    assert extract_methods("public class X {\n}") == []
    assert extract_methods("final class P { private int a; }") == []


def test_unparseable_snippet_raises_a_parse_error():
    with pytest.raises(PARSE_ERRORS):
        extract_methods("public class Broken { void f( { }")
