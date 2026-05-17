import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def sym_solve(equation: str, variable: str = "x") -> str:
        """Solve an equation symbolically. Pass expression (set equal to 0) or 'Eq(lhs, rhs)' syntax. Example: 'x**2 - 4' solves x^2 = 4."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            if equation.strip().startswith("Eq("):
                expr = sympy.sympify(equation, locals=local_vars)
            else:
                expr = safe_parse_expr(equation, local_vars)
            solutions = sympy.solve(expr, var)
            return str(solutions)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_simplify(expression: str) -> str:
        """Simplify a symbolic math expression. Example: 'sin(x)**2 + cos(x)**2' returns '1'."""
        try:
            expr = safe_parse_expr(expression)
            return str(sympy.simplify(expr))
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_expand(expression: str) -> str:
        """Expand a symbolic expression (distribute, multiply out). Example: '(x+1)**3'."""
        try:
            expr = safe_parse_expr(expression)
            return str(sympy.expand(expr))
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_factor(expression: str) -> str:
        """Factor a symbolic expression. Example: 'x**2 - 4' returns '(x-2)*(x+2)'."""
        try:
            expr = safe_parse_expr(expression)
            return str(sympy.factor(expr))
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
        """Compute the nth derivative of an expression. Example: sym_differentiate('x**3', 'x', 2) returns '6*x'."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.diff(expr, var, order)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_integrate(expression: str, variable: str = "x", lower_bound: str = "", upper_bound: str = "") -> str:
        """Compute indefinite or definite integral. Leave bounds empty for indefinite. Bounds can be symbolic like 'oo' for infinity."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            if lower_bound and upper_bound:
                lo = safe_parse_expr(lower_bound)
                hi = safe_parse_expr(upper_bound)
                result = sympy.integrate(expr, (var, lo, hi))
            else:
                result = sympy.integrate(expr, var)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_limit(expression: str, variable: str = "x", point: str = "0", direction: str = "+") -> str:
        """Compute limit of expression as variable approaches point. Direction: '+' (right), '-' (left), or '+-' (both)."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            pt = safe_parse_expr(point)
            result = sympy.limit(expr, var, pt, direction)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def sym_series(expression: str, variable: str = "x", point: str = "0", order: int = 6) -> str:
        """Taylor/Laurent series expansion around a point to given order. Example: sym_series('exp(x)', 'x', '0', 5)."""
        try:
            var = sympy.Symbol(variable)
            local_vars = {variable: var}
            expr = safe_parse_expr(expression, local_vars)
            pt = safe_parse_expr(point)
            result = sympy.series(expr, var, pt, order)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
