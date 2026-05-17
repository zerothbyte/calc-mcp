import json
import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def groebner_basis(polynomials: str, variables: str = "x,y,z", order: str = "grevlex") -> str:
        """Compute Gröbner basis of a system of polynomials. polynomials: semicolon-separated polynomial expressions. variables: comma-separated. order: grevlex, lex, grlex. Example: groebner_basis('x**2 + y - 1; x + y**2 - 1', 'x,y')."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}

            polys = [safe_parse_expr(p.strip(), local_vars) for p in polynomials.split(";")]
            basis = sympy.groebner(polys, *syms, order=order)
            return json.dumps({
                "basis": [str(b) for b in basis],
                "order": order,
                "variables": vars_list,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def polynomial_gcd(poly_a: str, poly_b: str, variable: str = "x") -> str:
        """Compute GCD of two polynomials. Example: polynomial_gcd('x**3 - 1', 'x**2 - 1', 'x')."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            a = safe_parse_expr(poly_a, local_vars)
            b = safe_parse_expr(poly_b, local_vars)
            gcd = sympy.gcd(a, b, var)
            return json.dumps({"gcd": str(gcd), "poly_a": poly_a, "poly_b": poly_b})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def polynomial_resultant(poly_a: str, poly_b: str, variable: str = "x") -> str:
        """Compute resultant of two polynomials (eliminates variable). Resultant is 0 iff polynomials share a common root. Example: polynomial_resultant('x**2 + a*x + 1', 'x + a', 'x')."""
        try:
            var = sympy.Symbol(variable)
            a_sym = sympy.Symbol("a")
            b_sym = sympy.Symbol("b")
            local_vars = {variable: var, "a": a_sym, "b": b_sym}
            p = safe_parse_expr(poly_a, local_vars)
            q = safe_parse_expr(poly_b, local_vars)
            res = sympy.resultant(p, q, var)
            return json.dumps({"resultant": str(res), "variable_eliminated": variable})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def polynomial_factor_over_field(polynomial: str, variable: str = "x", domain: str = "ZZ") -> str:
        """Factor a polynomial over a specified domain. Domains: ZZ (integers), QQ (rationals), RR (reals), CC (complex), GF(p) (finite field). Example: polynomial_factor_over_field('x**4 - 1', 'x', 'CC')."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            poly = safe_parse_expr(polynomial, local_vars)

            if domain == "ZZ":
                result = sympy.factor(poly, domain=sympy.ZZ)
            elif domain == "QQ":
                result = sympy.factor(poly, domain=sympy.QQ)
            elif domain == "RR":
                result = sympy.factor(poly, domain=sympy.RR)
            elif domain == "CC":
                result = sympy.factor(poly, gaussian=True)
            elif domain.startswith("GF(") and domain.endswith(")"):
                p = int(domain[3:-1])
                result = sympy.factor(poly, modulus=p)
            else:
                return f"Error: Unknown domain '{domain}'. Use: ZZ, QQ, RR, CC, GF(p)"

            return json.dumps({"factored": str(result), "domain": domain})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def solve_polynomial_system(equations: str, variables: str = "x,y") -> str:
        """Solve a system of polynomial equations using Gröbner basis elimination. equations: semicolon-separated. Returns all solutions."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}

            polys = [safe_parse_expr(p.strip(), local_vars) for p in equations.split(";")]
            solutions = sympy.solve(polys, syms, dict=True)
            result = [{str(k): str(v) for k, v in sol.items()} for sol in solutions]
            return json.dumps({"solutions": result, "count": len(result)})
        except Exception as e:
            return f"Error: {e}"
