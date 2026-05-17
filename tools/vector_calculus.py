import sympy
from tools.utils import safe_parse_expr


def register_tools(mcp):

    @mcp.tool()
    def gradient(expression: str, variables: str = "x,y,z") -> str:
        """Compute gradient (nabla f) of a scalar field. variables: comma-separated variable names. Example: gradient('x**2 + y**2 + z**2', 'x,y,z')."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            expr = safe_parse_expr(expression, local_vars)
            grad = [str(sympy.diff(expr, s)) for s in syms]
            return str(grad)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def divergence(field: str, variables: str = "x,y,z") -> str:
        """Compute divergence of a vector field. field: semicolon-separated components (Fx;Fy;Fz). Example: divergence('x**2; y**2; z**2', 'x,y,z')."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            components = [safe_parse_expr(c.strip(), local_vars) for c in field.split(";")]
            if len(components) != len(syms):
                return f"Error: Number of components ({len(components)}) must match variables ({len(syms)})"
            div = sum(sympy.diff(comp, var) for comp, var in zip(components, syms))
            return str(div)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def curl(field: str, variables: str = "x,y,z") -> str:
        """Compute curl of a 3D vector field. field: semicolon-separated components (Fx;Fy;Fz). Example: curl('-y; x; 0', 'x,y,z')."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            if len(vars_list) != 3:
                return "Error: Curl is defined for 3D vector fields only"
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            components = [safe_parse_expr(c.strip(), local_vars) for c in field.split(";")]
            if len(components) != 3:
                return "Error: Need exactly 3 components for curl"
            x, y, z = syms
            Fx, Fy, Fz = components
            curl_result = [
                str(sympy.diff(Fz, y) - sympy.diff(Fy, z)),
                str(sympy.diff(Fx, z) - sympy.diff(Fz, x)),
                str(sympy.diff(Fy, x) - sympy.diff(Fx, y)),
            ]
            return str(curl_result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def laplacian(expression: str, variables: str = "x,y,z") -> str:
        """Compute Laplacian (div(grad f)) of a scalar field. Example: laplacian('x**2 + y**2 + z**2', 'x,y,z')."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            expr = safe_parse_expr(expression, local_vars)
            lap = sum(sympy.diff(expr, s, 2) for s in syms)
            return str(lap)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def jacobian_matrix(fields: str, variables: str = "x,y,z") -> str:
        """Compute Jacobian matrix of a vector field. fields: semicolon-separated expressions. variables: comma-separated. Returns matrix as string."""
        try:
            vars_list = [v.strip() for v in variables.split(",")]
            syms = [sympy.Symbol(v) for v in vars_list]
            local_vars = {v: s for v, s in zip(vars_list, syms)}
            components = [safe_parse_expr(c.strip(), local_vars) for c in fields.split(";")]
            jac = sympy.Matrix([[sympy.diff(f, v) for v in syms] for f in components])
            return str(jac.tolist())
        except Exception as e:
            return f"Error: {e}"
