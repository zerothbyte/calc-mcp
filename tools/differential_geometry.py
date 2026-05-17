import json
import sympy
from tools.utils import safe_parse_expr


def _parse_metric(metric: str, coordinates: str):
    """Parse metric string and validate format."""
    coords = [v.strip() for v in coordinates.split(",")]
    syms = [sympy.Symbol(c) for c in coords]
    local_vars = {c: s for c, s in zip(coords, syms)}
    n = len(coords)

    rows = [row.strip() for row in metric.split(";")]
    if len(rows) != n:
        raise ValueError(
            f"Invalid metric format: got {len(rows)} rows but {n} coordinates ({coordinates}). "
            f"Ensure metric is a string with rows separated by ';' and elements by ','. "
            f"Example for 2D: '1,0;0,r**2' with coordinates 'r,theta'"
        )
    for i, row in enumerate(rows):
        elems = row.split(",")
        if len(elems) != n:
            raise ValueError(
                f"Invalid metric format: row {i+1} has {len(elems)} elements but expected {n}. "
                f"Each row must have exactly {n} comma-separated elements to match coordinates ({coordinates})."
            )

    g = sympy.Matrix(n, n, lambda i, j: safe_parse_expr(rows[i].split(",")[j].strip(), local_vars))
    return g, syms, local_vars, n, coords


def register_tools(mcp):

    @mcp.tool()
    def christoffel_symbols(metric: str, coordinates: str = "r,theta,phi") -> str:
        """Compute Christoffel symbols of the second kind from a metric tensor. metric: rows separated by ';', elements by ','. Example: '1,0;0,r**2' with coordinates 'r,theta'."""
        try:
            g, syms, local_vars, n, coords = _parse_metric(metric, coordinates)
            g_inv = g.inv()

            christoffel = {}
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(mu, n):
                        val = sympy.Rational(1, 2) * sum(
                            g_inv[sigma, lam] * (
                                sympy.diff(g[lam, nu], syms[mu]) +
                                sympy.diff(g[lam, mu], syms[nu]) -
                                sympy.diff(g[mu, nu], syms[lam])
                            ) for lam in range(n)
                        )
                        val = sympy.simplify(val)
                        if val != 0:
                            key = f"Gamma^{coords[sigma]}_{coords[mu]}{coords[nu]}"
                            christoffel[key] = str(val)

            return json.dumps(christoffel) if christoffel else json.dumps({"message": "All Christoffel symbols are zero (flat space)"})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def riemann_tensor(metric: str, coordinates: str = "r,theta") -> str:
        """Compute non-zero components of the Riemann curvature tensor R^sigma_mu_nu_rho from a metric. metric: rows separated by ';', elements by ','."""
        try:
            g, syms, local_vars, n, coords = _parse_metric(metric, coordinates)
            g_inv = g.inv()

            def gamma(sigma, mu, nu):
                return sympy.Rational(1, 2) * sum(
                    g_inv[sigma, lam] * (
                        sympy.diff(g[lam, nu], syms[mu]) +
                        sympy.diff(g[lam, mu], syms[nu]) -
                        sympy.diff(g[mu, nu], syms[lam])
                    ) for lam in range(n)
                )

            riemann = {}
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        for rho in range(n):
                            val = (sympy.diff(gamma(sigma, mu, rho), syms[nu])
                                   - sympy.diff(gamma(sigma, mu, nu), syms[rho])
                                   + sum(gamma(sigma, nu, lam) * gamma(lam, mu, rho) for lam in range(n))
                                   - sum(gamma(sigma, rho, lam) * gamma(lam, mu, nu) for lam in range(n)))
                            val = sympy.simplify(val)
                            if val != 0:
                                key = f"R^{coords[sigma]}_{coords[mu]}{coords[nu]}{coords[rho]}"
                                riemann[key] = str(val)

            return json.dumps(riemann) if riemann else json.dumps({"message": "All Riemann tensor components are zero (flat space)"})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def ricci_tensor_and_scalar(metric: str, coordinates: str = "r,theta") -> str:
        """Compute Ricci tensor R_mu_nu and Ricci scalar R from a metric. metric: rows separated by ';', elements by ','."""
        try:
            g, syms, local_vars, n, coords = _parse_metric(metric, coordinates)
            g_inv = g.inv()

            def gamma(sigma, mu, nu):
                return sympy.Rational(1, 2) * sum(
                    g_inv[sigma, lam] * (
                        sympy.diff(g[lam, nu], syms[mu]) +
                        sympy.diff(g[lam, mu], syms[nu]) -
                        sympy.diff(g[mu, nu], syms[lam])
                    ) for lam in range(n)
                )

            ricci = {}
            ricci_matrix = sympy.zeros(n, n)
            for mu in range(n):
                for nu in range(n):
                    val = sum(
                        sympy.diff(gamma(lam, mu, nu), syms[lam])
                        - sympy.diff(gamma(lam, mu, lam), syms[nu])
                        + sum(gamma(lam, lam, sig) * gamma(sig, mu, nu) for sig in range(n))
                        - sum(gamma(lam, nu, sig) * gamma(sig, mu, lam) for sig in range(n))
                        for lam in range(n)
                    )
                    val = sympy.simplify(val)
                    ricci_matrix[mu, nu] = val
                    if val != 0:
                        ricci[f"R_{coords[mu]}{coords[nu]}"] = str(val)

            scalar = sympy.simplify(sum(g_inv[mu, nu] * ricci_matrix[mu, nu] for mu in range(n) for nu in range(n)))

            result = {"ricci_tensor": ricci if ricci else "zero", "ricci_scalar": str(scalar)}
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def surface_curvature(parametrization: str, parameters: str = "u,v") -> str:
        """Compute Gaussian and mean curvature of a parametric surface. parametrization: semicolon-separated x;y;z expressions in parameters. Example: 'cos(u)*cos(v); cos(u)*sin(v); sin(u)' for a sphere."""
        try:
            params = [p.strip() for p in parameters.split(",")]
            syms = [sympy.Symbol(p) for p in params]
            local_vars = {p: s for p, s in zip(params, syms)}

            components = [safe_parse_expr(c.strip(), local_vars) for c in parametrization.split(";")]
            if len(components) != 3:
                return "Error: Need exactly 3 components (x;y;z)"

            u, v = syms[0], syms[1]
            r = sympy.Matrix(components)
            r_u = r.diff(u)
            r_v = r.diff(v)
            r_uu = r_u.diff(u)
            r_uv = r_u.diff(v)
            r_vv = r_v.diff(v)

            E = r_u.dot(r_u)
            F = r_u.dot(r_v)
            G = r_v.dot(r_v)

            normal = r_u.cross(r_v)
            n_mag = sympy.sqrt(normal.dot(normal))
            n_hat = normal / n_mag

            L = r_uu.dot(n_hat)
            M = r_uv.dot(n_hat)
            N = r_vv.dot(n_hat)

            K = sympy.simplify((L*N - M**2) / (E*G - F**2))
            H = sympy.simplify((E*N + G*L - 2*F*M) / (2*(E*G - F**2)))

            return json.dumps({
                "gaussian_curvature": str(K),
                "mean_curvature": str(H),
                "first_fundamental_form": {"E": str(E), "F": str(F), "G": str(G)},
                "second_fundamental_form": {"L": str(L), "M": str(M), "N": str(N)},
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def geodesic_equations(metric: str, coordinates: str = "r,theta", parameter: str = "s") -> str:
        """Derive geodesic equations from a metric tensor. Returns the system of ODEs for geodesics. metric: rows separated by ';', elements by ','."""
        try:
            g, syms, local_vars, n, coords = _parse_metric(metric, coordinates)
            s = sympy.Symbol(parameter)
            g_inv = g.inv()

            equations = {}
            for sigma in range(n):
                rhs_terms = []
                for mu in range(n):
                    for nu in range(n):
                        gamma_val = sympy.Rational(1, 2) * sum(
                            g_inv[sigma, lam] * (
                                sympy.diff(g[lam, nu], syms[mu]) +
                                sympy.diff(g[lam, mu], syms[nu]) -
                                sympy.diff(g[mu, nu], syms[lam])
                            ) for lam in range(n)
                        )
                        gamma_val = sympy.simplify(gamma_val)
                        if gamma_val != 0:
                            rhs_terms.append(f"-{gamma_val}*d{coords[mu]}/d{parameter}*d{coords[nu]}/d{parameter}")
                eq = " + ".join(rhs_terms) if rhs_terms else "0"
                equations[f"d^2{coords[sigma]}/d{parameter}^2"] = eq

            return json.dumps(equations)
        except Exception as e:
            return f"Error: {e}"
