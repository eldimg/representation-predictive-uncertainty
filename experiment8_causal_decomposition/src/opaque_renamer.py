from __future__ import annotations

import ast
import io
import symtable
import tokenize
from dataclasses import dataclass
from typing import Any


OPAQUE_STEMS = (
    "zav",
    "mip",
    "tuk",
    "qel",
    "vor",
    "nex",
    "bim",
    "rud",
    "kaf",
    "wex",
    "pov",
    "jil",
    "sud",
    "fyn",
    "gop",
    "laz",
)


@dataclass(frozen=True)
class RenameResult:
    source: str
    mapping: dict[str, str]
    canonical_function_name: str
    opaque_function_name: str
    canonical_locals: tuple[str, ...]
    opaque_locals: tuple[str, ...]
    canonical_globals: tuple[str, ...]
    opaque_globals: tuple[str, ...]


def _single_function(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected exactly one top-level FunctionDef")
    function = tree.body[0]
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        and node is not function
        for node in ast.walk(function)
    ):
        raise ValueError("Nested functions/lambdas are unsupported by opacity renamer")
    return function


def _function_symbols(source: str) -> tuple[set[str], set[str]]:
    table = symtable.symtable(source, "<opaque-check>", "exec")
    children = [child for child in table.get_children() if child.get_type() == "function"]
    if len(children) != 1:
        raise ValueError("Expected exactly one function symbol table")
    child = children[0]
    locals_ = {
        symbol.get_name()
        for symbol in child.get_symbols()
        if symbol.is_local() or symbol.is_parameter()
    }
    globals_ = {
        symbol.get_name()
        for symbol in child.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    return locals_, globals_


def _ordered_bound_names(function: ast.FunctionDef) -> list[str]:
    names = [function.name]
    names.extend(argument.arg for argument in function.args.args)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
    seen: set[str] = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def _opaque_name(index: int) -> str:
    stem = OPAQUE_STEMS[index % len(OPAQUE_STEMS)]
    cycle = index // len(OPAQUE_STEMS)
    return f"{stem}{cycle}"


def build_mapping(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    function = _single_function(tree)
    ordered = _ordered_bound_names(function)
    occupied = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    occupied.add(function.name)
    mapping: dict[str, str] = {}
    opaque_index = 0
    for original in ordered:
        while True:
            candidate = _opaque_name(opaque_index)
            opaque_index += 1
            if candidate not in occupied and candidate not in mapping.values():
                break
        mapping[original] = candidate
    return mapping


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute_offset(source: str, offsets: list[int], line: int, column: int) -> int:
    line_text = source.splitlines(keepends=True)[line - 1]
    prefix = line_text.encode("utf-8")[:column].decode("utf-8")
    return offsets[line - 1] + len(prefix)


def _replacement_spans(source: str, mapping: dict[str, str]) -> list[tuple[int, int, str]]:
    tree = ast.parse(source)
    function = _single_function(tree)
    offsets = _line_offsets(source)
    replacements: set[tuple[int, int, str]] = set()

    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id in mapping:
            start = _absolute_offset(source, offsets, node.lineno, node.col_offset)
            end = _absolute_offset(source, offsets, node.end_lineno, node.end_col_offset)
            replacements.add((start, end, mapping[node.id]))
        elif isinstance(node, ast.arg) and node.arg in mapping:
            start = _absolute_offset(source, offsets, node.lineno, node.col_offset)
            end = start + len(node.arg)
            replacements.add((start, end, mapping[node.arg]))

    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    saw_def = False
    for token in tokens:
        if token.type == tokenize.NAME and token.string == "def":
            saw_def = True
            continue
        if saw_def and token.type == tokenize.NAME:
            if token.string != function.name:
                raise ValueError("Could not identify FunctionDef name token")
            start = _absolute_offset(source, offsets, token.start[0], token.start[1])
            end = _absolute_offset(source, offsets, token.end[0], token.end[1])
            replacements.add((start, end, mapping[function.name]))
            break

    return sorted(replacements, reverse=True)


def _apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    output = source
    last_start = len(source) + 1
    for start, end, replacement in replacements:
        if end > last_start:
            raise ValueError("Overlapping identifier replacement spans")
        output = output[:start] + replacement + output[end:]
        last_start = start
    return output


class _NameNormalizer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> Any:
        node.arg = self.mapping.get(node.arg, node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        node.id = self.mapping.get(node.id, node.id)
        return node


def normalized_ast_equivalent(
    canonical_source: str,
    opaque_source: str,
    mapping: dict[str, str],
) -> bool:
    canonical_tree = ast.parse(canonical_source)
    opaque_tree = ast.parse(opaque_source)
    placeholders = {
        original: f"__BOUND_{index:03d}"
        for index, original in enumerate(mapping)
    }
    reverse_placeholders = {
        opaque: placeholders[original] for original, opaque in mapping.items()
    }
    canonical_tree = _NameNormalizer(placeholders).visit(canonical_tree)
    opaque_tree = _NameNormalizer(reverse_placeholders).visit(opaque_tree)
    ast.fix_missing_locations(canonical_tree)
    ast.fix_missing_locations(opaque_tree)
    return ast.dump(canonical_tree, include_attributes=False) == ast.dump(
        opaque_tree, include_attributes=False
    )


def rename_program(source: str) -> RenameResult:
    mapping = build_mapping(source)
    canonical_tree = ast.parse(source)
    canonical_function = _single_function(canonical_tree)
    opaque_source = _apply_replacements(source, _replacement_spans(source, mapping))
    opaque_tree = ast.parse(opaque_source)
    opaque_function = _single_function(opaque_tree)
    compile(canonical_tree, "<canonical>", "exec")
    compile(opaque_tree, "<opaque>", "exec")

    canonical_locals, canonical_globals = _function_symbols(source)
    opaque_locals, opaque_globals = _function_symbols(opaque_source)
    expected_opaque_locals = {mapping.get(name, name) for name in canonical_locals}
    if opaque_locals != expected_opaque_locals:
        raise ValueError(
            f"Opaque local binding mismatch: {opaque_locals} != {expected_opaque_locals}"
        )
    if opaque_globals != canonical_globals:
        raise ValueError(
            f"Global/API symbol mismatch: {opaque_globals} != {canonical_globals}"
        )
    if not normalized_ast_equivalent(source, opaque_source, mapping):
        raise ValueError("Normalized AST mismatch after identifier opacity transform")

    return RenameResult(
        source=opaque_source,
        mapping=mapping,
        canonical_function_name=canonical_function.name,
        opaque_function_name=opaque_function.name,
        canonical_locals=tuple(sorted(canonical_locals)),
        opaque_locals=tuple(sorted(opaque_locals)),
        canonical_globals=tuple(sorted(canonical_globals)),
        opaque_globals=tuple(sorted(opaque_globals)),
    )
