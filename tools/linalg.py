import json
import numpy as np
from scipy import stats as scipy_stats
from tools.utils import parse_json_array, parse_json_matrix


def register_tools(mcp):

    @mcp.tool()
    def matrix_determinant(matrix: str) -> str:
        """Compute determinant of a square matrix. Input: JSON 2D array, e.g. '[[1,2],[3,4]]'."""
        try:
            data = parse_json_matrix(matrix)
            m = np.array(data, dtype=float)
            if m.shape[0] != m.shape[1]:
                return "Error: Matrix must be square"
            det = np.linalg.det(m)
            return str(round(det, 10))
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_inverse(matrix: str) -> str:
        """Compute inverse of a square matrix. Input: JSON 2D array."""
        try:
            data = parse_json_matrix(matrix)
            m = np.array(data, dtype=float)
            if m.shape[0] != m.shape[1]:
                return "Error: Matrix must be square"
            inv = np.linalg.inv(m)
            return json.dumps(inv.tolist())
        except np.linalg.LinAlgError:
            return "Error: Matrix is singular (not invertible)"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_eigenvalues(matrix: str) -> str:
        """Compute eigenvalues and eigenvectors of a square matrix. Input: JSON 2D array."""
        try:
            data = parse_json_matrix(matrix)
            m = np.array(data, dtype=float)
            if m.shape[0] != m.shape[1]:
                return "Error: Matrix must be square"
            eigenvalues, eigenvectors = np.linalg.eig(m)
            result = {
                "eigenvalues": eigenvalues.tolist(),
                "eigenvectors": eigenvectors.tolist(),
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def matrix_multiply(matrix_a: str, matrix_b: str) -> str:
        """Multiply two matrices. Inputs: JSON 2D arrays."""
        try:
            a = np.array(parse_json_matrix(matrix_a), dtype=float)
            b = np.array(parse_json_matrix(matrix_b), dtype=float)
            if a.shape[1] != b.shape[0]:
                return f"Error: Incompatible dimensions {a.shape} x {b.shape}"
            result = (a @ b).tolist()
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def solve_linear_system(coefficients: str, constants: str) -> str:
        """Solve linear system Ax = b. coefficients: JSON 2D array (A), constants: JSON 1D array (b)."""
        try:
            A = np.array(parse_json_matrix(coefficients), dtype=float)
            b = np.array(parse_json_array(constants), dtype=float)
            if A.shape[0] != A.shape[1]:
                return "Error: Coefficient matrix must be square"
            if A.shape[0] != b.shape[0]:
                return "Error: Dimensions of A and b don't match"
            x = np.linalg.solve(A, b)
            return json.dumps(x.tolist())
        except np.linalg.LinAlgError:
            return "Error: System is singular or has no unique solution"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def descriptive_stats(data: str) -> str:
        """Compute descriptive statistics: mean, median, std, min, max, Q1, Q3, skewness, kurtosis. Input: JSON 1D array."""
        try:
            arr = np.array(parse_json_array(data), dtype=float)
            result = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q1": float(np.percentile(arr, 25)),
                "q3": float(np.percentile(arr, 75)),
                "skewness": float(scipy_stats.skew(arr)),
                "kurtosis": float(scipy_stats.kurtosis(arr)),
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"
