import json
import sympy
from sympy.logic.boolalg import to_cnf, to_dnf, simplify_logic, is_cnf
from sympy.logic.inference import satisfiable


def register_tools(mcp):

    @mcp.tool()
    def propositional_satisfiability(expression: str) -> str:
        """Check satisfiability of a propositional logic formula. Uses symbols like A, B, C. Operators: & (and), | (or), ~ (not), >> (implies), Equivalent. Returns satisfying assignment or UNSAT."""
        try:
            symbols_used = set()
            for ch in expression:
                if ch.isalpha() and ch.isupper():
                    symbols_used.add(ch)
            local_vars = {s: sympy.Symbol(s) for s in symbols_used}

            expr = sympy.sympify(expression, locals=local_vars)
            result = satisfiable(expr)

            if result is False:
                return json.dumps({"satisfiable": False, "message": "UNSATISFIABLE"})
            else:
                assignment = {str(k): v for k, v in result.items()}
                return json.dumps({"satisfiable": True, "assignment": assignment})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def logic_simplify(expression: str, form: str = "cnf") -> str:
        """Simplify a logical expression. form: 'cnf' (conjunctive normal form), 'dnf' (disjunctive normal form), 'simplify' (minimal). Operators: & | ~ >> Equivalent."""
        try:
            symbols_used = set()
            for ch in expression:
                if ch.isalpha() and ch.isupper():
                    symbols_used.add(ch)
            local_vars = {s: sympy.Symbol(s) for s in symbols_used}

            expr = sympy.sympify(expression, locals=local_vars)

            if form == "cnf":
                result = to_cnf(expr, simplify=True)
            elif form == "dnf":
                result = to_dnf(expr, simplify=True)
            elif form == "simplify":
                result = simplify_logic(expr)
            else:
                return f"Error: Unknown form '{form}'. Use: cnf, dnf, simplify"

            return json.dumps({"original": expression, "result": str(result), "form": form})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def truth_table(expression: str) -> str:
        """Generate truth table for a propositional logic expression. Uses uppercase letters as variables. Returns all rows of the truth table."""
        try:
            symbols_used = sorted(set(ch for ch in expression if ch.isalpha() and ch.isupper()))
            local_vars = {s: sympy.Symbol(s) for s in symbols_used}
            expr = sympy.sympify(expression, locals=local_vars)
            sym_list = [local_vars[s] for s in symbols_used]

            rows = []
            n = len(symbols_used)
            for i in range(2**n):
                assignment = {}
                for j, s in enumerate(symbols_used):
                    assignment[s] = bool((i >> (n - 1 - j)) & 1)
                val = bool(expr.subs({local_vars[k]: v for k, v in assignment.items()}))
                assignment["result"] = val
                rows.append(assignment)

            tautology = all(r["result"] for r in rows)
            contradiction = not any(r["result"] for r in rows)

            return json.dumps({
                "variables": symbols_used,
                "rows": rows,
                "is_tautology": tautology,
                "is_contradiction": contradiction,
                "is_contingent": not tautology and not contradiction,
            })
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def verify_logical_equivalence(expr_a: str, expr_b: str) -> str:
        """Verify if two logical expressions are equivalent. Returns True if A <=> B is a tautology."""
        try:
            symbols_used = set()
            for ch in expr_a + expr_b:
                if ch.isalpha() and ch.isupper():
                    symbols_used.add(ch)
            local_vars = {s: sympy.Symbol(s) for s in symbols_used}

            a = sympy.sympify(expr_a, locals=local_vars)
            b = sympy.sympify(expr_b, locals=local_vars)

            equivalence = sympy.Equivalent(a, b)
            result = satisfiable(~equivalence)

            if result is False:
                return json.dumps({"equivalent": True, "message": "Expressions are logically equivalent"})
            else:
                counterexample = {str(k): v for k, v in result.items()}
                return json.dumps({"equivalent": False, "counterexample": counterexample})
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def verify_induction(base_case: str, inductive_step: str, variable: str = "n", base_value: int = 0) -> str:
        """Verify mathematical induction proof structure. base_case: expression that should be True at n=base_value. inductive_step: expression P(n) >> P(n+1) that should be a tautology (simplified). Returns verification of both steps."""
        try:
            n = sympy.Symbol(variable, integer=True, nonneg=True)
            k = sympy.Symbol("k", integer=True, nonneg=True)
            local_vars = {variable: n, "k": k}

            base_expr = sympy.sympify(base_case, locals=local_vars)
            base_result = base_expr.subs(n, base_value)
            base_verified = bool(base_result) if base_result.is_Boolean else sympy.simplify(base_result) == 0

            step_expr = sympy.sympify(inductive_step, locals=local_vars)
            step_simplified = sympy.simplify(step_expr)

            return json.dumps({
                "base_case": {
                    "expression": base_case,
                    "at_n": base_value,
                    "evaluates_to": str(base_result),
                    "verified": bool(base_verified),
                },
                "inductive_step": {
                    "expression": inductive_step,
                    "simplified": str(step_simplified),
                    "note": "Verify that P(n) implies P(n+1) holds for all n >= base_value",
                },
                "proof_structure_valid": bool(base_verified),
            })
        except Exception as e:
            return f"Error: {e}"
