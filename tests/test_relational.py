"""Hand-checked ATFD / LAA / FDP on a small Java file."""
import math

from phase_a_data.relational import analyze_source

SRC = """
package com.example;

import java.util.List;

public class Report extends Base {
    private int total;
    private Customer customer;
    public static final int LIMIT = 5;

    public Report(int total) { this.total = total; }

    public int score(Order order, Address addr) {
        int t = total + this.total;                 // own: total
        String city = addr.getCity();                // foreign Address.city
        String zip = addr.zip;                        // foreign Address.zip
        int q = order.getQuantity() + order.qty;      // foreign Order.quantity, Order.qty
        String n = customer.getName();               // foreign Customer.name
        String s = order.getItem().getLabel();       // foreign Order.item + chained ?getItem().label
        List<String> xs = null; xs.size();           // ignored type, not accessor
        int m = Math.max(LIMIT, Config.MAX_SIZE);    // constants not counted
        setTotal(t);                                  // own accessor -> total
        return q;
    }

    public void setTotal(int v) { total = v; }

    static class Inner {
        int depth;
        int peek(Report r) { return r.total + depth; }  // Report is enclosing, not own type here
    }
}
"""

def test_relational():
    classes, methods = analyze_source(SRC, "/src/Report.java", ["String", "List", "Math"])
    by_cls = {c["class_fqn"]: c for c in classes}
    assert set(by_cls) == {"com.example.Report", "com.example.Report.Inner"}
    rep = by_cls["com.example.Report"]
    # Address.city, Address.zip, Order.quantity, Order.qty, Order.item, Customer.name, ?getItem().label
    assert rep["atfd"] == 7, rep
    assert rep["fdp"] == 4, rep          # Address, Order, Customer, ?getItem()
    assert rep["parse_error"] is None
    m = {(x["method"], x["params"]): x for x in methods}
    score = m[("score", "Order,Address")]
    assert score["atfd"] == 7 and score["n_own_attrs"] == 1   # LIMIT is a constant, not counted
    assert math.isclose(score["laa"], 1 / 8)
    ctor = m[("Report", "int")]
    assert ctor["is_constructor"] and ctor["atfd"] == 0 and ctor["n_own_attrs"] == 1
    inner = by_cls["com.example.Report.Inner"]
    assert inner["atfd"] == 1            # Report.total


DATA = """
public class Point {
    public int x;
    public int y;
    public static int COUNT;
    private String label;
    public Point() {}
    public String getLabel() { return label; }
    public void setLabel(String l) { label = l; }
    public boolean isOrigin() { return x == 0 && y == 0; }
    public double norm() { return Math.sqrt(x * x + y * y); }
    private void helper() {}
    public static Point zero() { return new Point(); }
}
"""


def test_data_class_inputs():
    classes, _ = analyze_source(DATA, "/Point.java")
    c = classes[0]
    # NOPA: x, y (COUNT static). Public non-static methods: getLabel, setLabel, isOrigin, norm.
    assert c["nopa"] == 2 and c["n_public_methods"] == 4 and c["noam"] == 3
    assert math.isclose(c["woc"], 1 / 6)     # norm / (4 methods + 2 attrs)


def test_parse_error_is_reported():
    classes, methods = analyze_source("class {", "/x/Bad.java")
    assert len(classes) == 1 and classes[0]["parse_error"] and methods == []


ARRAY_REF_SRC = """
package p;
public class A {
    private Node node;
    public B[] all(java.util.List<C> cs) {
        return cs.stream().map(c -> { B b = c.getB(); return b; }).toArray(B[]::new);
    }
    public int size() { return node.count; }
}
"""


def test_array_constructor_reference_is_parsed():
    """javalang 0.13 rejects `B[]::new`; the retry blanks `[]` and the metrics match the
    equivalent `n -> new B[n]` lambda."""
    from phase_a_data.relational import blank_array_ctor_refs
    blanked = blank_array_ctor_refs(ARRAY_REF_SRC)
    assert len(blanked) == len(ARRAY_REF_SRC) and "B  ::new" in blanked
    classes, methods = analyze_source(ARRAY_REF_SRC, "/p/A.java")
    assert classes[0]["parse_error"] is None and classes[0]["atfd"] == 2   # C.b, Node.count
    same, _ = analyze_source(ARRAY_REF_SRC.replace("B[]::new", "n -> new B[n]"), "/p/A.java")
    assert classes == same
    assert [m["line"] for m in methods] == [5, 8]   # lines unchanged by the retry


def test_type_use_annotation_in_type_arguments_is_parsed():
    """javalang 0.13 rejects `Map<K, @Nullable V>`; the retry blanks the annotation and the
    metrics match the same source without it."""
    src = """
package p;
import java.util.*;
public class S {
    private Map<Integer, @Nullable List<Packet>> pending = new HashMap<>();
    public int n(Packet q) { return q.size + pending.size(); }
}
"""
    classes, _ = analyze_source(src, "/p/S.java")
    assert classes[0]["parse_error"] is None and classes[0]["atfd"] == 1   # Packet.size
    same, _ = analyze_source(src.replace("@Nullable ", ""), "/p/S.java")
    assert classes == same


def test_unfixable_parse_error_keeps_original_message():
    classes, _ = analyze_source("class A { int[] f() { return x.toArray(int[]::new) } }", "/A.java")
    assert classes[0]["parse_error"].startswith("JavaSyntaxError")
