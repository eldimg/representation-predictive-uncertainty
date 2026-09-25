from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterator


FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.Yield,
    ast.YieldFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

SUPPORTED_STATEMENTS = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.While,
    ast.If,
    ast.Return,
    ast.Expr,
    ast.Break,
    ast.Continue,
    ast.Pass,
)


def ast_to_dict(node: Any) -> Any:
    if isinstance(node, ast.AST):
        value: dict[str, Any] = {"_type": type(node).__name__}
        for field in node._fields:
            value[field] = ast_to_dict(getattr(node, field))
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                value[attr] = getattr(node, attr)
        return value
    if isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    return node


def dict_to_ast(value: Any) -> Any:
    if isinstance(value, list):
        return [dict_to_ast(item) for item in value]
    if not isinstance(value, dict) or "_type" not in value:
        return value
    node_type = getattr(ast, value["_type"])
    kwargs = {
        field: dict_to_ast(value.get(field))
        for field in getattr(node_type, "_fields", ())
    }
    node = node_type(**kwargs)
    for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
        if attr in value and value[attr] is not None:
            setattr(node, attr, value[attr])
    return node


def path_key(path: tuple[Any, ...] | list[Any]) -> str:
    return "/".join(str(part) for part in path)


def iter_ast_with_paths(
    node: ast.AST,
    path: tuple[Any, ...] = (),
) -> Iterator[tuple[ast.AST, tuple[Any, ...]]]:
    yield node, path
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from iter_ast_with_paths(value, path + (field,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield from iter_ast_with_paths(item, path + (field, index))


def get_node_at_path(root: ast.AST, path: list[Any] | tuple[Any, ...]) -> ast.AST:
    current: Any = root
    index = 0
    while index < len(path):
        part = path[index]
        if isinstance(part, str):
            current = getattr(current, part)
        else:
            current = current[part]
        index += 1
    if not isinstance(current, ast.AST):
        raise TypeError(f"Path does not resolve to AST node: {path}")
    return current


def validate_program_shape(tree: ast.Module, trajectory_id: str) -> ast.FunctionDef:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"{trajectory_id}: expected one top-level FunctionDef")
    function = tree.body[0]
    if function.decorator_list:
        raise ValueError(f"{trajectory_id}: decorators are unsupported")
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise ValueError(f"{trajectory_id}: variadic/keyword-only parameters unsupported")
    if function.args.defaults or function.args.kw_defaults:
        raise ValueError(f"{trajectory_id}: default arguments unsupported")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(
                f"{trajectory_id}: forbidden AST node {type(node).__name__}"
            )
        if isinstance(node, ast.stmt) and node is not function:
            if not isinstance(node, SUPPORTED_STATEMENTS):
                raise ValueError(
                    f"{trajectory_id}: unsupported statement {type(node).__name__}"
                )
    return function


@dataclass
class ProgramIndex:
    trace_nodes: list[dict[str, Any]]
    semantic_steps: list[dict[str, Any]]


def index_program(tree: ast.Module, source: str) -> ProgramIndex:
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    trace_nodes: list[dict[str, Any]] = []
    semantic_steps: list[dict[str, Any]] = []
    trace_counter = 0
    step_counter = 0

    def next_trace() -> str:
        nonlocal trace_counter
        trace_counter += 1
        return f"N{trace_counter:04d}"

    def next_step(kind: str, path: tuple[Any, ...], trace_id: str) -> str:
        nonlocal step_counter
        step_counter += 1
        step_id = f"S{step_counter:03d}"
        semantic_steps.append(
            {
                "semantic_step_id": step_id,
                "kind": kind,
                "path": list(path),
                "path_key": path_key(path),
                "trace_node_id": trace_id,
            }
        )
        return step_id

    def visit(node: ast.AST, path: tuple[Any, ...], owner: str | None) -> None:
        trace_id = next_trace()
        if isinstance(node, ast.FunctionDef):
            owner = next_step("FUNCTION_INTERFACE", path, trace_id)
        elif isinstance(node, ast.stmt):
            owner = next_step(type(node).__name__.upper(), path, trace_id)
        trace_nodes.append(
            {
                "trace_node_id": trace_id,
                "node_type": type(node).__name__,
                "path": list(path),
                "path_key": path_key(path),
                "owner_semantic_step_id": owner,
            }
        )
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, path + (field,), owner)
            elif isinstance(value, list):
                for item_index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        visit(item, path + (field, item_index), owner)

    visit(function, (), None)
    return ProgramIndex(trace_nodes=trace_nodes, semantic_steps=semantic_steps)


def build_program_ir(
    task: dict[str, Any], variant_index: int, source: str
) -> dict[str, Any]:
    trajectory_id = f"{task['id']}__v{variant_index + 1}"
    tree = ast.parse(source)
    function = validate_program_shape(tree, trajectory_id)
    index = index_program(tree, source)
    return {
        "program_id": trajectory_id,
        "task_id": task["id"],
        "family": task["family"],
        "variant": variant_index + 1,
        "task_text": task["task"],
        "source_python": source,
        "source_nl": task["nl"][variant_index],
        "interface": {
            "function_name": function.name,
            "parameters": [argument.arg for argument in function.args.args],
        },
        "body": {
            "statement_count": sum(
                1
                for node in ast.walk(function)
                if isinstance(node, ast.stmt) and node is not function
            )
        },
        "ast": ast_to_dict(tree),
        "trace_nodes": index.trace_nodes,
        "semantic_steps": index.semantic_steps,
    }
