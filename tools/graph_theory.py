import json
import numpy as np
from tools.utils import parse_json_matrix, parse_json_array


def register_tools(mcp):

    @mcp.tool()
    def max_flow(capacity_matrix: str, source: int = 0, sink: int = -1) -> str:
        """Compute maximum flow in a network using Edmonds-Karp (BFS-based Ford-Fulkerson). capacity_matrix: JSON adjacency matrix of capacities. source/sink: node indices."""
        try:
            cap = parse_json_matrix(capacity_matrix)
            n = len(cap)
            if sink == -1:
                sink = n - 1
            capacity = [row[:] for row in cap]

            def bfs(source, sink, parent):
                visited = [False] * n
                visited[source] = True
                queue = [source]
                while queue:
                    u = queue.pop(0)
                    for v in range(n):
                        if not visited[v] and capacity[u][v] > 0:
                            visited[v] = True
                            parent[v] = u
                            if v == sink:
                                return True
                            queue.append(v)
                return False

            max_flow_val = 0
            parent = [-1] * n

            while bfs(source, sink, parent):
                path_flow = float('inf')
                v = sink
                while v != source:
                    u = parent[v]
                    path_flow = min(path_flow, capacity[u][v])
                    v = parent[v]

                v = sink
                while v != source:
                    u = parent[v]
                    capacity[u][v] -= path_flow
                    capacity[v][u] += path_flow
                    v = parent[v]

                max_flow_val += path_flow
                parent = [-1] * n

            return json.dumps({"max_flow": max_flow_val, "source": source, "sink": sink})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def min_spanning_tree(adjacency: str) -> str:
        """Compute Minimum Spanning Tree using Kruskal's algorithm. adjacency: JSON adjacency matrix (0 or null = no edge). Returns edges and total weight."""
        try:
            matrix = parse_json_matrix(adjacency)
            n = len(matrix)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i][j] and matrix[i][j] > 0:
                        edges.append((matrix[i][j], i, j))
            edges.sort()

            parent = list(range(n))
            rank = [0] * n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True

            mst_edges = []
            total_weight = 0
            for w, u, v in edges:
                if union(u, v):
                    mst_edges.append({"from": u, "to": v, "weight": w})
                    total_weight += w
                    if len(mst_edges) == n - 1:
                        break

            return json.dumps({"edges": mst_edges, "total_weight": total_weight, "num_edges": len(mst_edges)})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def graph_chromatic_number(adjacency: str) -> str:
        """Estimate chromatic number (minimum colors for proper vertex coloring) using greedy algorithm. adjacency: JSON adjacency matrix (1=edge, 0=no edge). Returns coloring and chromatic number upper bound."""
        try:
            matrix = parse_json_matrix(adjacency)
            n = len(matrix)
            colors = [-1] * n

            for node in range(n):
                neighbor_colors = set()
                for adj in range(n):
                    if matrix[node][adj] and colors[adj] != -1:
                        neighbor_colors.add(colors[adj])
                color = 0
                while color in neighbor_colors:
                    color += 1
                colors[node] = color

            chromatic = max(colors) + 1
            return json.dumps({
                "chromatic_number_upper_bound": chromatic,
                "coloring": colors,
                "note": "Greedy upper bound; exact chromatic number is NP-hard",
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def topological_sort(adjacency: str) -> str:
        """Topological sort of a directed acyclic graph (DAG). adjacency: JSON adjacency matrix (1=edge from row to col). Returns sorted order or error if cycle exists."""
        try:
            matrix = parse_json_matrix(adjacency)
            n = len(matrix)
            in_degree = [0] * n
            for i in range(n):
                for j in range(n):
                    if matrix[i][j]:
                        in_degree[j] += 1

            queue = [i for i in range(n) if in_degree[i] == 0]
            result = []

            while queue:
                queue.sort()
                node = queue.pop(0)
                result.append(node)
                for j in range(n):
                    if matrix[node][j]:
                        in_degree[j] -= 1
                        if in_degree[j] == 0:
                            queue.append(j)

            if len(result) != n:
                return json.dumps({"error": "Graph contains a cycle, topological sort not possible"})
            return json.dumps({"topological_order": result})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def graph_characteristic_polynomial(adjacency: str, variable: str = "x") -> str:
        """Compute characteristic polynomial of a graph's adjacency matrix. adjacency: JSON adjacency matrix. Returns polynomial in variable."""
        try:
            import sympy
            matrix = parse_json_matrix(adjacency)
            M = sympy.Matrix(matrix)
            x = sympy.Symbol(variable)
            char_poly = M.charpoly(x)
            return json.dumps({
                "characteristic_polynomial": str(char_poly.as_expr()),
                "eigenvalues": [str(e) for e in M.eigenvals().keys()],
            })
        except Exception as e:
            return f"Error: {e}"
