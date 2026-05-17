import json
import sympy
import numpy as np


def register_tools(mcp):

    @mcp.tool()
    def combinatorics(n: int, r: int, operation: str = "combination") -> str:
        """Compute combinatorial values. Operations: combination (nCr), permutation (nPr), multichoose (stars and bars), derangement, catalan, bell, stirling2. For catalan/bell/derangement only n is used."""
        try:
            if operation == "combination":
                result = int(sympy.binomial(n, r))
            elif operation == "permutation":
                result = int(sympy.factorial(n) / sympy.factorial(n - r))
            elif operation == "multichoose":
                result = int(sympy.binomial(n + r - 1, r))
            elif operation == "derangement":
                result = int(sympy.subfactorial(n))
            elif operation == "catalan":
                result = int(sympy.catalan(n))
            elif operation == "bell":
                result = int(sympy.bell(n))
            elif operation == "stirling2":
                result = int(sympy.functions.combinatorial.numbers.stirling(n, r, kind=2))
            else:
                return f"Error: Unknown operation '{operation}'. Use: combination, permutation, multichoose, derangement, catalan, bell, stirling2"
            return json.dumps({"operation": operation, "n": n, "r": r, "result": result})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def modular_arithmetic(a: int, b: int, modulus: int, operation: str = "power") -> str:
        """Modular arithmetic operations. Operations: power (a^b mod m), inverse (modular inverse of a mod m), add, multiply, order (multiplicative order)."""
        try:
            if operation == "power":
                result = pow(a, b, modulus)
            elif operation == "inverse":
                result = pow(a, -1, modulus)
            elif operation == "add":
                result = (a + b) % modulus
            elif operation == "multiply":
                result = (a * b) % modulus
            elif operation == "order":
                result = int(sympy.n_order(a, modulus))
            else:
                return f"Error: Unknown operation '{operation}'"
            return json.dumps({"a": a, "b": b, "modulus": modulus, "operation": operation, "result": result})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def number_theory(n: int, operation: str = "factorize") -> str:
        """Number theory operations. Operations: factorize, is_prime, next_prime, euler_totient, divisors, mobius, legendre (needs second arg as n2 in JSON)."""
        try:
            if operation == "factorize":
                result = dict(sympy.factorint(n))
                return json.dumps({"n": n, "factors": result, "factorization": str(sympy.factorint(n, visual=True))})
            elif operation == "is_prime":
                result = sympy.isprime(n)
                return json.dumps({"n": n, "is_prime": result})
            elif operation == "next_prime":
                result = int(sympy.nextprime(n))
                return json.dumps({"n": n, "next_prime": result})
            elif operation == "euler_totient":
                result = int(sympy.totient(n))
                return json.dumps({"n": n, "euler_totient": result})
            elif operation == "divisors":
                divs = sorted([int(d) for d in sympy.divisors(n)])
                return json.dumps({"n": n, "divisors": divs, "count": len(divs), "sum": sum(divs)})
            elif operation == "mobius":
                result = int(sympy.mobius(n))
                return json.dumps({"n": n, "mobius": result})
            else:
                return f"Error: Unknown operation '{operation}'"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def gcd_lcm(numbers: str) -> str:
        """Compute GCD and LCM of a list of integers. Input: JSON array of integers. Also returns prime factorizations."""
        try:
            from tools.utils import parse_json_array
            nums = [int(x) for x in parse_json_array(numbers)]
            from math import gcd
            from functools import reduce
            g = reduce(gcd, nums)
            lcm_val = reduce(lambda a, b: a * b // gcd(a, b), nums)
            return json.dumps({"numbers": nums, "gcd": g, "lcm": lcm_val})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def graph_shortest_path(adjacency: str, start: int = 0, end: int = -1) -> str:
        """Find shortest path in a weighted graph using Dijkstra. adjacency: JSON adjacency matrix (use 0 or null for no edge). start/end: node indices. If end=-1, returns distances to all nodes."""
        try:
            from tools.utils import parse_json_matrix
            matrix = parse_json_matrix(adjacency)
            n = len(matrix)
            if end == -1:
                end = n - 1

            dist = [float('inf')] * n
            dist[start] = 0
            visited = [False] * n
            prev = [-1] * n

            for _ in range(n):
                u = -1
                for v in range(n):
                    if not visited[v] and (u == -1 or dist[v] < dist[u]):
                        u = v
                if dist[u] == float('inf'):
                    break
                visited[u] = True
                for v in range(n):
                    w = matrix[u][v]
                    if w and w > 0 and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = u

            path = []
            node = end
            while node != -1:
                path.append(node)
                node = prev[node]
            path.reverse()

            return json.dumps({
                "start": start,
                "end": end,
                "distance": dist[end] if dist[end] != float('inf') else None,
                "path": path if dist[end] != float('inf') else [],
                "all_distances": [d if d != float('inf') else None for d in dist],
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def generate_primes(lower: int = 2, upper: int = 100) -> str:
        """Generate all prime numbers in range [lower, upper]. Also returns count and sum."""
        try:
            primes = list(sympy.primerange(lower, upper + 1))
            return json.dumps({"primes": primes, "count": len(primes), "sum": sum(primes), "range": [lower, upper]})
        except Exception as e:
            return f"Error: {e}"
