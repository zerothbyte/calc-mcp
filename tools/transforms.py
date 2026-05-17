import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def fourier_transform(expression: str, variable: str = "x", transform_var: str = "k") -> str:
        """Compute symbolic Fourier transform F(k) = integral of f(x)*exp(-2*pi*i*k*x) dx. Example: fourier_transform('exp(-x**2)', 'x', 'k')."""
        try:
            x = sympy.Symbol(variable)
            k = sympy.Symbol(transform_var)
            local_vars = {variable: x, transform_var: k}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.fourier_transform(expr, x, k)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def inverse_fourier_transform(expression: str, variable: str = "k", transform_var: str = "x") -> str:
        """Compute symbolic inverse Fourier transform. Example: inverse_fourier_transform('exp(-pi*k**2)', 'k', 'x')."""
        try:
            k = sympy.Symbol(variable)
            x = sympy.Symbol(transform_var)
            local_vars = {variable: k, transform_var: x}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.inverse_fourier_transform(expr, k, x)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def laplace_transform(expression: str, variable: str = "t", transform_var: str = "s") -> str:
        """Compute symbolic Laplace transform L{f(t)} = F(s). Example: laplace_transform('exp(-a*t)*sin(b*t)', 't', 's')."""
        try:
            t = sympy.Symbol(variable, positive=True)
            s = sympy.Symbol(transform_var)
            a = sympy.Symbol("a")
            b = sympy.Symbol("b")
            local_vars = {variable: t, transform_var: s, "a": a, "b": b}
            expr = safe_parse_expr(expression, local_vars)
            result, convergence, _ = sympy.laplace_transform(expr, t, s)
            return f"{result}  (converges for: {convergence})"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def inverse_laplace_transform(expression: str, variable: str = "s", transform_var: str = "t") -> str:
        """Compute symbolic inverse Laplace transform L^-1{F(s)} = f(t). Example: inverse_laplace_transform('1/(s**2 + 1)', 's', 't')."""
        try:
            s = sympy.Symbol(variable)
            t = sympy.Symbol(transform_var, positive=True)
            local_vars = {variable: s, transform_var: t}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.inverse_laplace_transform(expr, s, t)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def mellin_transform(expression: str, variable: str = "x", transform_var: str = "s") -> str:
        """Compute symbolic Mellin transform M{f(x)} = integral of x^(s-1)*f(x) dx from 0 to inf. Example: mellin_transform('exp(-x)', 'x', 's')."""
        try:
            x = sympy.Symbol(variable, positive=True)
            s = sympy.Symbol(transform_var)
            local_vars = {variable: x, transform_var: s}
            expr = safe_parse_expr(expression, local_vars)
            result, strip, _ = sympy.mellin_transform(expr, x, s)
            return f"{result}  (fundamental strip: {strip})"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def z_transform(expression: str, variable: str = "n", transform_var: str = "z") -> str:
        """Compute Z-transform symbolically: Z{f(n)} = sum of f(n)*z^(-n) for n=0 to inf. Example: z_transform('a**n', 'n', 'z') for geometric sequence."""
        try:
            n = sympy.Symbol(variable, integer=True, nonneg=True)
            z = sympy.Symbol(transform_var)
            a = sympy.Symbol("a")
            local_vars = {variable: n, transform_var: z, "a": a}
            expr = safe_parse_expr(expression, local_vars)
            result = sympy.summation(expr * z**(-n), (n, 0, sympy.oo))
            return str(sympy.simplify(result))
        except Exception as e:
            return f"Error: {e}"
