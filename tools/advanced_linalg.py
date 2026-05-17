import json
import numpy as np
import sympy
from tools.utils import parse_json_matrix


def register_tools(mcp):

    @mcp.tool()
    def matrix_svd(matrix: str) -> str:
        """Compute Singular Value Decomposition (SVD). Returns U, S (singular values), and Vt matrices. Input: JSON 2D array."""
        try:
            m = np.array(parse_json_matrix(matrix), dtype=float)
            U, S, Vt = np.linalg.svd(m)
            return json.dumps({"U": U.tolist(), "singular_values": S.tolist(), "Vt": Vt.tolist()})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_qr(matrix: str) -> str:
        """Compute QR decomposition. Returns orthogonal Q and upper triangular R. Input: JSON 2D array."""
        try:
            m = np.array(parse_json_matrix(matrix), dtype=float)
            Q, R = np.linalg.qr(m)
            return json.dumps({"Q": Q.tolist(), "R": R.tolist()})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_lu(matrix: str) -> str:
        """Compute LU decomposition (with pivoting). Returns P, L, U such that PA = LU. Input: JSON 2D square array."""
        try:
            from scipy.linalg import lu
            m = np.array(parse_json_matrix(matrix), dtype=float)
            P, L, U = lu(m)
            return json.dumps({"P": P.tolist(), "L": L.tolist(), "U": U.tolist()})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_cholesky(matrix: str) -> str:
        """Compute Cholesky decomposition. Matrix must be symmetric positive-definite. Returns lower triangular L such that A = L @ L.T. Input: JSON 2D array."""
        try:
            m = np.array(parse_json_matrix(matrix), dtype=float)
            L = np.linalg.cholesky(m)
            return json.dumps({"L": L.tolist()})
        except np.linalg.LinAlgError:
            return "Error: Matrix is not positive-definite"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_jordan_form(matrix: str) -> str:
        """Compute Jordan normal form of a matrix (symbolic/exact). Input: JSON 2D square array. Returns Jordan matrix J and transformation P such that A = P*J*P^-1."""
        try:
            data = parse_json_matrix(matrix)
            m = sympy.Matrix(data)
            P, J = m.jordan_form()
            return json.dumps({"J": str(J.tolist()), "P": str(P.tolist())})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_rank(matrix: str) -> str:
        """Compute rank of a matrix. Input: JSON 2D array."""
        try:
            m = np.array(parse_json_matrix(matrix), dtype=float)
            rank = int(np.linalg.matrix_rank(m))
            return json.dumps({"rank": rank, "shape": list(m.shape)})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_null_space(matrix: str) -> str:
        """Compute null space (kernel) of a matrix. Returns basis vectors for the null space. Input: JSON 2D array."""
        try:
            m = sympy.Matrix(parse_json_matrix(matrix))
            ns = m.nullspace()
            result = [list(map(str, vec)) for vec in ns]
            return json.dumps({"null_space_basis": result, "nullity": len(ns)})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_exp(matrix: str) -> str:
        """Compute matrix exponential e^A. Useful for solving systems of linear ODEs. Input: JSON 2D square array."""
        try:
            from scipy.linalg import expm
            m = np.array(parse_json_matrix(matrix), dtype=float)
            result = expm(m)
            return json.dumps({"matrix_exp": result.tolist()})
        except Exception as e:
            return f"Error: {e}"
