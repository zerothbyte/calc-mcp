import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def complex_residue(expression: str, variable: str = "z", point: str = "0") -> str:
        """Compute the residue of a complex function at a given point. Example: complex_residue('1/(z*(z-1))', 'z', '0') returns 1."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            pt = safe_parse_expr(point, local_vars)
            res = sympy.residue(expr, var, pt)
            return str(res)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def complex_partial_fractions(expression: str, variable: str = "z") -> str:
        """Decompose a rational function into partial fractions. Example: complex_partial_fractions('1/(z**2 - 1)', 'z')."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.apart(expr, var)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def complex_laurent_series(expression: str, variable: str = "z", point: str = "0", order: int = 6) -> str:
        """Compute Laurent series of a complex function around a point. Includes negative powers (principal part). Example: complex_laurent_series('exp(1/z)', 'z', '0', 5)."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            pt = safe_parse_expr(point)

            # For essential singularities, substitute w = 1/(z - pt) to get Taylor series then convert back
            shifted = var - pt if pt != 0 else var
            # Try direct series first
            series = sympy.series(expr, var, pt, n=order)
            result_str = str(series)

            # If series didn't expand (returned the expression itself), try substitution approach
            if result_str == str(expr) or 'O(' not in result_str:
                w = sympy.Symbol("_w")
                substituted = expr.subs(var, 1/w + pt) if pt != 0 else expr.subs(var, 1/w)
                taylor = sympy.series(substituted, w, 0, n=order)
                back = taylor.subs(w, 1/shifted)
                result_str = str(back)

            return result_str
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def complex_contour_integral(expression: str, variable: str = "z", singularities: str = "[]") -> str:
        """Evaluate a contour integral using the residue theorem (assumes closed contour enclosing given singularities). singularities: JSON array of points inside contour. Result = 2*pi*i * sum(residues)."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)

            import json
            points = json.loads(singularities)
            total_residue = 0
            for pt_str in points:
                pt = safe_parse_expr(str(pt_str))
                total_residue += sympy.residue(expr, var, pt)

            result = 2 * sympy.pi * sympy.I * total_residue
            return str(sympy.simplify(result))
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def complex_poles(expression: str, variable: str = "z") -> str:
        """Find poles of a complex function and their orders. Returns list of {pole, order, residue} objects."""
        try:
            import json as json_mod
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)

            numer, denom = sympy.fraction(sympy.together(expr))
            poles = sympy.solve(denom, var)

            result = []
            for pole in poles:
                order = 1
                for n in range(1, 10):
                    test = sympy.limit((var - pole)**n * expr, var, pole)
                    if test.is_finite and test != 0:
                        order = n
                        break
                result.append({"pole": str(pole), "order": order, "residue": str(sympy.residue(expr, var, pole))})

            return json_mod.dumps(result)
        except Exception as e:
            return f"Error: {e}"
