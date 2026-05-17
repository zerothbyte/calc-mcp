import json
import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def series_convergence(expression: str, variable: str = "n") -> str:
        """Test convergence of an infinite series sum(a_n, n=1..inf). Applies ratio test, root test, and divergence test. expression: general term a_n. Example: '1/n**2'."""
        try:
            n = sympy.Symbol(variable, positive=True)
            local_vars = {variable: n}
            a_n = safe_parse_expr(expression, local_vars)

            results = {}

            try:
                ratio = sympy.limit(sympy.Abs(a_n.subs(n, n+1) / a_n), n, sympy.oo)
                ratio_val = float(ratio) if ratio.is_number else None
                if ratio_val is not None:
                    if ratio_val < 1:
                        results["ratio_test"] = {"limit": str(ratio), "conclusion": "converges absolutely"}
                    elif ratio_val > 1:
                        results["ratio_test"] = {"limit": str(ratio), "conclusion": "diverges"}
                    else:
                        results["ratio_test"] = {"limit": str(ratio), "conclusion": "inconclusive"}
                else:
                    results["ratio_test"] = {"limit": str(ratio), "conclusion": "inconclusive"}
            except Exception:
                results["ratio_test"] = {"conclusion": "could not compute"}

            try:
                root = sympy.limit(sympy.Abs(a_n)**(sympy.Rational(1, n)), n, sympy.oo)
                root_val = float(root) if root.is_number else None
                if root_val is not None:
                    if root_val < 1:
                        results["root_test"] = {"limit": str(root), "conclusion": "converges absolutely"}
                    elif root_val > 1:
                        results["root_test"] = {"limit": str(root), "conclusion": "diverges"}
                    else:
                        results["root_test"] = {"limit": str(root), "conclusion": "inconclusive"}
                else:
                    results["root_test"] = {"limit": str(root), "conclusion": "inconclusive"}
            except Exception:
                results["root_test"] = {"conclusion": "could not compute"}

            term_limit = sympy.limit(a_n, n, sympy.oo)
            if term_limit != 0:
                results["divergence_test"] = {"limit": str(term_limit), "conclusion": "diverges (term does not approach 0)"}
            else:
                results["divergence_test"] = {"limit": "0", "conclusion": "inconclusive (term approaches 0)"}

            try:
                total = sympy.summation(a_n, (n, 1, sympy.oo))
                results["sum"] = str(total)
            except Exception:
                results["sum"] = "could not compute"

            return json.dumps(results)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def power_series_radius(expression: str, variable: str = "n") -> str:
        """Find radius of convergence of a power series with general coefficient a_n. Uses ratio test: R = lim |a_n / a_(n+1)|. Example: '1/factorial(n)' gives R=inf (exp series)."""
        try:
            n = sympy.Symbol(variable, positive=True)
            local_vars = {variable: n}
            a_n = safe_parse_expr(expression, local_vars)

            ratio = sympy.limit(sympy.Abs(a_n / a_n.subs(n, n+1)), n, sympy.oo)
            radius = sympy.simplify(ratio)

            return json.dumps({"coefficient_a_n": expression, "radius_of_convergence": str(radius)})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def fourier_series(expression: str, variable: str = "x", period: str = "2*pi", num_terms: int = 5) -> str:
        """Compute Fourier series coefficients (a0, an, bn) of a periodic function. expression: one period of f(x). period: the period T. Returns coefficients and series representation."""
        try:
            x = sympy.Symbol(variable)
            local_vars = {variable: x}
            f = safe_parse_expr(expression, local_vars)
            T = safe_parse_expr(period, local_vars)
            L = T / 2

            a0 = sympy.simplify(sympy.integrate(f, (x, -L, L)) / T)

            an_list = []
            bn_list = []
            n = sympy.Symbol("n", positive=True, integer=True)

            for k in range(1, num_terms + 1):
                an = sympy.simplify(2 * sympy.integrate(f * sympy.cos(k * sympy.pi * x / L), (x, -L, L)) / T)
                bn = sympy.simplify(2 * sympy.integrate(f * sympy.sin(k * sympy.pi * x / L), (x, -L, L)) / T)
                an_list.append(str(an))
                bn_list.append(str(bn))

            series_terms = [f"{a0}"]
            for k in range(num_terms):
                if an_list[k] != "0":
                    series_terms.append(f"{an_list[k]}*cos({k+1}*pi*{variable}/{L})")
                if bn_list[k] != "0":
                    series_terms.append(f"{bn_list[k]}*sin({k+1}*pi*{variable}/{L})")

            return json.dumps({
                "a0": str(a0),
                "an": an_list,
                "bn": bn_list,
                "series": " + ".join(series_terms),
                "num_terms": num_terms,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sequence_limit(expression: str, variable: str = "n") -> str:
        """Compute the limit of a sequence as n -> infinity. Also checks if monotone and bounded. Example: '(1 + 1/n)**n' returns e."""
        try:
            n = sympy.Symbol(variable, positive=True)
            local_vars = {variable: n}
            a_n = safe_parse_expr(expression, local_vars)

            lim = sympy.limit(a_n, n, sympy.oo)

            a_n1 = a_n.subs(n, n + 1)
            diff = sympy.simplify(a_n1 - a_n)

            try:
                is_increasing = sympy.ask(sympy.Q.positive(diff))
                is_decreasing = sympy.ask(sympy.Q.negative(diff))
            except Exception:
                is_increasing = None
                is_decreasing = None

            monotonicity = "increasing" if is_increasing else ("decreasing" if is_decreasing else "unknown/not monotone")

            return json.dumps({
                "limit": str(lim),
                "converges": lim.is_finite if hasattr(lim, 'is_finite') else str(lim) != "oo",
                "monotonicity": monotonicity,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def integral_convergence(expression: str, variable: str = "x", lower: str = "1", upper: str = "oo") -> str:
        """Test convergence of an improper integral. Computes the integral if convergent. Example: integral_convergence('1/x**2', 'x', '1', 'oo')."""
        try:
            x = sympy.Symbol(variable)
            local_vars = {variable: x}
            f = safe_parse_expr(expression, local_vars)
            lo = safe_parse_expr(lower, local_vars)
            hi = safe_parse_expr(upper, local_vars)

            result = sympy.integrate(f, (x, lo, hi))
            is_convergent = result.is_finite if hasattr(result, 'is_finite') else (result != sympy.oo and result != -sympy.oo)

            return json.dumps({
                "integral": str(result),
                "converges": bool(is_convergent),
                "expression": expression,
                "bounds": [str(lo), str(hi)],
            })
        except Exception as e:
            return f"Error: {e}"
