from __future__ import annotations

import argparse
import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    CONDITIONS,
    IR_FILE,
    ORDER_FILE,
    ORDER_SEED,
    RENDERED_FILE,
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)
from ir import dict_to_ast, get_node_at_path, path_key
from opaque_renamer import rename_program


@dataclass
class Rendered:
    text: str
    trace_spans: list[dict[str, Any]]
    semantic_step_spans: list[dict[str, Any]]


class Emitter:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.step_spans: dict[str, dict[str, Any]] = {}

    def emit_line(self, text: str, step_id: str | None = None) -> None:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        end = self.length
        self.parts.append("\n")
        self.length += 1
        if step_id is not None:
            if step_id in self.step_spans:
                raise ValueError(f"Semantic step emitted twice: {step_id}")
            self.step_spans[step_id] = {
                "semantic_step_id": step_id,
                "start_char": start,
                "end_char": end,
            }

    def finish(self) -> str:
        return "".join(self.parts).rstrip("\n")


BIN_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitOr: "|",
    ast.BitAnd: "&",
}
COMPARE_OPS = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "IN",
    ast.NotIn: "NOT IN",
    ast.Is: "IS",
    ast.IsNot: "IS NOT",
}


class ExpressionRenderer:
    def __init__(self, style: str):
        self.style = style

    def render(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if node.value is True:
                return "true" if self.style == "prose" else "TRUE"
            if node.value is False:
                return "false" if self.style == "prose" else "FALSE"
            if node.value is None:
                return "nothing" if self.style == "prose" else "NONE"
            return repr(node.value)
        if isinstance(node, ast.BinOp):
            operator = BIN_OPS.get(type(node.op), type(node.op).__name__.upper())
            return f"({self.render(node.left)} {operator} {self.render(node.right)})"
        if isinstance(node, ast.UnaryOp):
            operator = {
                ast.Not: "not" if self.style == "prose" else "NOT ",
                ast.USub: "-",
                ast.UAdd: "+",
                ast.Invert: "~",
            }.get(type(node.op), type(node.op).__name__.upper())
            spacer = " " if operator.isalpha() else ""
            return f"{operator}{spacer}{self.render(node.operand)}"
        if isinstance(node, ast.BoolOp):
            operator = " and " if isinstance(node.op, ast.And) else " or "
            if self.style != "prose":
                operator = operator.upper()
            return "(" + operator.join(self.render(value) for value in node.values) + ")"
        if isinstance(node, ast.Compare):
            parts = [self.render(node.left)]
            for operator, comparator in zip(node.ops, node.comparators):
                token = COMPARE_OPS.get(type(operator), type(operator).__name__.upper())
                if self.style == "prose":
                    token = {
                        "=": "equals",
                        "!=": "does not equal",
                        "<": "is less than",
                        "<=": "is at most",
                        ">": "is greater than",
                        ">=": "is at least",
                        "IN": "is in",
                        "NOT IN": "is not in",
                        "IS": "is",
                        "IS NOT": "is not",
                    }.get(token, token)
                parts.extend([token, self.render(comparator)])
            return " ".join(parts)
        if isinstance(node, ast.Call):
            args = [self.render(argument) for argument in node.args]
            args.extend(
                f"{keyword.arg}={self.render(keyword.value)}"
                for keyword in node.keywords
                if keyword.arg is not None
            )
            return f"{self.render(node.func)}({', '.join(args)})"
        if isinstance(node, ast.Attribute):
            return f"{self.render(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{self.render(node.value)}[{self.render(node.slice)}]"
        if isinstance(node, ast.Slice):
            lower = self.render(node.lower)
            upper = self.render(node.upper)
            step = self.render(node.step)
            return f"{lower}:{upper}" + (f":{step}" if node.step is not None else "")
        if isinstance(node, ast.List):
            return "[" + ", ".join(self.render(item) for item in node.elts) + "]"
        if isinstance(node, ast.Tuple):
            body = ", ".join(self.render(item) for item in node.elts)
            if len(node.elts) == 1:
                body += ","
            return f"({body})"
        if isinstance(node, ast.Dict):
            pairs = [
                f"{self.render(key)}: {self.render(value)}"
                for key, value in zip(node.keys, node.values)
            ]
            return "{" + ", ".join(pairs) + "}"
        if isinstance(node, ast.IfExp):
            if self.style == "prose":
                return (
                    f"{self.render(node.body)} if {self.render(node.test)}, "
                    f"otherwise {self.render(node.orelse)}"
                )
            return (
                f"IF {self.render(node.test)} THEN {self.render(node.body)} "
                f"ELSE {self.render(node.orelse)}"
            )
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class StructuredRenderer:
    def __init__(self, program: dict[str, Any], style: str):
        self.program = program
        self.style = style
        self.tree = dict_to_ast(program["ast"])
        assert isinstance(self.tree, ast.Module)
        ast.fix_missing_locations(self.tree)
        self.function = self.tree.body[0]
        assert isinstance(self.function, ast.FunctionDef)
        self.expression = ExpressionRenderer(style)
        self.emitter = Emitter()
        self.step_by_path = {
            step["path_key"]: step["semantic_step_id"]
            for step in program["semantic_steps"]
        }

    def _step(self, path: tuple[Any, ...]) -> str:
        return self.step_by_path[path_key(path)]

    def _target(self, node: ast.AST) -> str:
        return self.expression.render(node)

    def _emit_interface(self) -> None:
        name = self.function.name
        parameters = [argument.arg for argument in self.function.args.args]
        step = self._step(())
        if self.style == "prose":
            if len(parameters) == 1:
                line = f"Define a procedure named {name} that receives {parameters[0]}."
            else:
                line = (
                    f"Define a procedure named {name} that receives "
                    + ", ".join(parameters[:-1])
                    + f" and {parameters[-1]}."
                )
        elif self.style == "controlled":
            line = f"PROCEDURE {name} INPUT " + ", ".join(parameters) + ":"
        else:
            line = f"PROCEDURE {name}({', '.join(parameters)})"
        self.emitter.emit_line(line, step)

    def _simple_line(self, node: ast.stmt) -> str:
        expression = self.expression.render
        if isinstance(node, ast.Assign):
            target = " = ".join(self._target(item) for item in node.targets)
            value = expression(node.value)
            if self.style == "prose":
                return f"Set {target} to {value}."
            if self.style == "controlled":
                return f"SET {target} TO {value}."
            return f"{target} <- {value}"
        if isinstance(node, ast.AnnAssign):
            target = self._target(node.target)
            value = expression(node.value)
            if self.style == "prose":
                return f"Set {target} to {value}."
            if self.style == "controlled":
                return f"SET {target} TO {value}."
            return f"{target} <- {value}"
        if isinstance(node, ast.AugAssign):
            target = self._target(node.target)
            value = expression(node.value)
            operator = BIN_OPS.get(type(node.op), type(node.op).__name__.upper())
            if self.style == "prose":
                verb = {
                    "+": "Increase",
                    "-": "Decrease",
                    "*": "Multiply",
                    "/": "Divide",
                }.get(operator)
                if verb == "Increase":
                    return f"Increase {target} by {value}."
                if verb == "Decrease":
                    return f"Decrease {target} by {value}."
                if verb == "Multiply":
                    return f"Multiply {target} by {value}."
                if verb == "Divide":
                    return f"Divide {target} by {value}."
                return f"Update {target} to {target} {operator} {value}."
            if self.style == "controlled":
                return f"UPDATE {target} USING {operator} {value}."
            return f"{target} <- {target} {operator} {value}"
        if isinstance(node, ast.Return):
            value = expression(node.value)
            return f"Return {value}." if self.style == "prose" else f"RETURN {value}"
        if isinstance(node, ast.Expr):
            value = expression(node.value)
            if self.style == "prose":
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "append"
                    and len(node.value.args) == 1
                ):
                    return (
                        f"Append {expression(node.value.args[0])} to "
                        f"{expression(node.value.func.value)}."
                    )
                return f"Perform {value}."
            if self.style == "controlled":
                return f"EXECUTE {value}."
            return value
        if isinstance(node, ast.Break):
            return "Stop the current loop." if self.style == "prose" else "BREAK"
        if isinstance(node, ast.Continue):
            return "Continue with the next iteration." if self.style == "prose" else "CONTINUE"
        if isinstance(node, ast.Pass):
            return "Do nothing." if self.style == "prose" else "NO OPERATION"
        raise ValueError(f"Unsupported simple statement: {type(node).__name__}")

    def _render_block(
        self,
        statements: list[ast.stmt],
        base_path: tuple[Any, ...],
        indent: int,
    ) -> None:
        for index, statement in enumerate(statements):
            self._render_statement(statement, base_path + (index,), indent)

    def _render_statement(
        self, node: ast.stmt, path: tuple[Any, ...], indent: int
    ) -> None:
        prefix = "    " * indent
        step = self._step(path)
        expression = self.expression.render
        if isinstance(node, ast.For):
            target = self._target(node.target)
            iterable = expression(node.iter)
            if self.style == "prose":
                line = f"For each {target} in {iterable}, do the following."
            elif self.style == "controlled":
                line = f"FOR EACH {target} IN {iterable}:"
            else:
                line = f"FOR {target} IN {iterable}"
            self.emitter.emit_line(prefix + line, step)
            self._render_block(node.body, path + ("body",), indent + 1)
            if node.orelse:
                self.emitter.emit_line(prefix + ("Otherwise." if self.style == "prose" else "ELSE"))
                self._render_block(node.orelse, path + ("orelse",), indent + 1)
            return
        if isinstance(node, ast.While):
            condition = expression(node.test)
            if self.style == "prose":
                line = f"While {condition}, do the following."
            elif self.style == "controlled":
                line = f"WHILE {condition}:"
            else:
                line = f"WHILE {condition}"
            self.emitter.emit_line(prefix + line, step)
            self._render_block(node.body, path + ("body",), indent + 1)
            if node.orelse:
                self.emitter.emit_line(prefix + ("After the loop, do the following." if self.style == "prose" else "ELSE"))
                self._render_block(node.orelse, path + ("orelse",), indent + 1)
            return
        if isinstance(node, ast.If):
            condition = expression(node.test)
            if self.style == "prose":
                line = f"If {condition}, do the following."
            elif self.style == "controlled":
                line = f"IF {condition}:"
            else:
                line = f"IF {condition}"
            self.emitter.emit_line(prefix + line, step)
            self._render_block(node.body, path + ("body",), indent + 1)
            if node.orelse:
                self.emitter.emit_line(prefix + ("Otherwise, do the following." if self.style == "prose" else "ELSE"))
                self._render_block(node.orelse, path + ("orelse",), indent + 1)
            return
        self.emitter.emit_line(prefix + self._simple_line(node), step)

    def render(self) -> Rendered:
        self._emit_interface()
        self._render_block(self.function.body, ("body",), 0)
        text = self.emitter.finish()
        step_spans = list(self.emitter.step_spans.values())
        trace_spans = []
        for trace in self.program["trace_nodes"]:
            owner = trace["owner_semantic_step_id"]
            span = self.emitter.step_spans.get(owner)
            if span is None:
                raise ValueError(f"Missing owner span for {trace['trace_node_id']}")
            trace_spans.append(
                {
                    "trace_node_id": trace["trace_node_id"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                }
            )
        return Rendered(text, trace_spans, step_spans)


def _line_starts(source: str) -> tuple[list[str], list[int]]:
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return lines, starts


def _source_offset(lines: list[str], starts: list[int], lineno: int, col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8")
    return starts[lineno - 1] + len(prefix)


def _step_line_start(lines: list[str], starts: list[int], lineno: int) -> int:
    """Include the immediately preceding line separator in the next step span.

    Code tokenizers often fuse ``\n`` and indentation with the first lexical token
    of a statement. Whitespace-only tokens remain analytically ineligible, while a
    fused whitespace+code token can map wholly to the following semantic step.
    """
    start = starts[lineno - 1]
    if lineno <= 1:
        return start
    previous = lines[lineno - 2]
    newline_length = len(previous) - len(previous.rstrip("\r\n"))
    return start - newline_length


def python_source_spans(program: dict[str, Any], source: str) -> Rendered:
    tree = ast.parse(source)
    lines, starts = _line_starts(source)
    step_spans: dict[str, dict[str, Any]] = {}
    for step in program["semantic_steps"]:
        node = get_node_at_path(tree.body[0], step["path"])
        if isinstance(node, ast.FunctionDef):
            start = _step_line_start(lines, starts, node.lineno)
            line_text = lines[node.lineno - 1].rstrip("\r\n")
            end = starts[node.lineno - 1] + len(line_text)
        elif isinstance(node, (ast.For, ast.While, ast.If)):
            start = _step_line_start(lines, starts, node.lineno)
            line_text = lines[node.lineno - 1].rstrip("\r\n")
            end = starts[node.lineno - 1] + len(line_text)
        else:
            start = _step_line_start(lines, starts, node.lineno)
            end = _source_offset(lines, starts, node.end_lineno, node.end_col_offset)
        step_spans[step["semantic_step_id"]] = {
            "semantic_step_id": step["semantic_step_id"],
            "start_char": start,
            "end_char": end,
        }
    trace_spans = []
    for trace in program["trace_nodes"]:
        owner = trace["owner_semantic_step_id"]
        span = step_spans[owner]
        trace_spans.append(
            {
                "trace_node_id": trace["trace_node_id"],
                "start_char": span["start_char"],
                "end_char": span["end_char"],
            }
        )
    return Rendered(source, trace_spans, list(step_spans.values()))


def _record(
    program: dict[str, Any],
    condition: str,
    rendered: Rendered,
    *,
    opacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trajectory_id": f"{program['program_id']}__{condition}",
        "program_id": program["program_id"],
        "task_id": program["task_id"],
        "family": program["family"],
        "variant": program["variant"],
        "condition": condition,
        "task_text": program["task_text"],
        "reference_text": rendered.text,
        "source_python": program["source_python"],
        "trace_spans": rendered.trace_spans,
        "semantic_step_spans": rendered.semantic_step_spans,
        "opacity": opacity,
    }


def render_program(program: dict[str, Any]) -> list[dict[str, Any]]:
    empty = Rendered(program["source_nl"], [], [])
    prose = StructuredRenderer(program, "prose").render()
    controlled = StructuredRenderer(program, "controlled").render()
    pseudocode = StructuredRenderer(program, "pseudocode").render()
    canonical = python_source_spans(program, program["source_python"])
    renamed = rename_program(program["source_python"])
    opaque = python_source_spans(program, renamed.source)
    opacity = {
        "mapping": renamed.mapping,
        "canonical_function_name": renamed.canonical_function_name,
        "opaque_function_name": renamed.opaque_function_name,
        "canonical_locals": list(renamed.canonical_locals),
        "opaque_locals": list(renamed.opaque_locals),
        "canonical_globals": list(renamed.canonical_globals),
        "opaque_globals": list(renamed.opaque_globals),
    }
    by_condition = {
        "nl_exp7_anchor": empty,
        "procedural_prose": prose,
        "controlled_nl": controlled,
        "pseudocode": pseudocode,
        "python_canonical": canonical,
        "python_opaque": opaque,
    }
    return [
        _record(
            program,
            condition,
            by_condition[condition],
            opacity=opacity if condition == "python_opaque" else None,
        )
        for condition in CONDITIONS
    ]


def render_all(ir_payload: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for program in ir_payload["programs"]:
        records.extend(render_program(program))
    if len(records) != 1800:
        raise RuntimeError(f"Expected 1800 rendered trajectories, found {len(records)}")
    payload = {
        "experiment": "Experiment 8 Sequential Representational Interventions",
        "schema_version": "2.0",
        "tasks_sha256": ir_payload["tasks_sha256"],
        "ir_content_sha256": ir_payload["content_sha256"],
        "trajectory_count": len(records),
        "records": records,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", default=str(IR_FILE))
    parser.add_argument("--output", default=str(RENDERED_FILE))
    args = parser.parse_args()
    ir_payload = load_json(Path(args.ir))
    rendered = render_all(ir_payload)
    output = Path(args.output)
    atomic_write_json(output, rendered)

    order = [record["trajectory_id"] for record in rendered["records"]]
    random.Random(ORDER_SEED).shuffle(order)
    atomic_write_json(ORDER_FILE, order)
    print(f"Wrote {len(rendered['records'])} trajectories to {output}")
    print(f"Rendered content SHA256: {rendered['content_sha256']}")
    print(f"File SHA256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
