from __future__ import annotations

import ast
import copy
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any


INTEGER_TASKS = {
    "factorial_iterative",
    "product_one_to_n",
    "fibonacci_nth",
    "fibonacci_first_n",
    "digit_sum",
    "digit_product",
    "count_digits",
    "count_even_digits",
    "reverse_integer",
    "numeric_palindrome",
    "is_prime",
    "count_primes_up_to",
    "list_divisors",
    "count_divisors",
    "integer_to_binary",
}
STRING_TASKS = {
    "count_vowels",
    "remove_digits",
    "keep_letters",
    "count_words",
    "count_unique_words",
    "string_palindrome",
    "reverse_string",
    "frequency_map_chars",
}
MATRIX_TASKS = {
    "transpose_matrix",
    "rotate_matrix_clockwise",
    "main_diagonal_sum",
    "anti_diagonal_sum",
    "matrix_row_sums",
    "matrix_column_sums",
}
LIST_OF_LISTS_TASKS = {"flatten_lists"}
PAIR_LIST_TASKS = {"flatten_pairs"}
STRING_LIST_TASKS = {"group_by_first_character", "group_by_length"}


def deterministic_cases(task_id: str) -> list[list[Any]]:
    if task_id in INTEGER_TASKS:
        return [[0], [1], [7], [12]]
    if task_id == "binary_to_integer":
        return [["0"], ["1"], ["101101"]]
    if task_id in STRING_TASKS:
        return [[""], ["level"], ["a1 b2  café"]]
    if task_id == "count_character":
        return [["banana", "a"], ["", "x"], ["mississippi", "s"]]
    if task_id in MATRIX_TASKS:
        return [
            [[[1]]],
            [[[1, 2], [3, 4]]],
            [[[1, 2, 3], [4, 5, 6]]],
        ]
    if task_id in {"matrix_contains", "matrix_find_position"}:
        return [
            [[[1, 2], [3, 4]], 3],
            [[[1, 2], [3, 4]], 9],
            [[[5]], 5],
        ]
    if task_id == "invert_unique_map":
        return [[{}], [{"a": 1, "b": 2}], [{"x": "u", "y": "v"}]]
    if task_id == "filter_mapping_by_value":
        return [[{}, 0], [{"a": 1, "b": 4}, 2], [{"x": -2, "y": 0}, -1]]
    if task_id in STRING_LIST_TASKS:
        return [[[]], [["a", "bee", "cat", "door"]], [["x", "xy", "z"]]]
    if task_id in LIST_OF_LISTS_TASKS:
        return [[[]], [[[1, 2], [], [3]]], [[[], [4, 5]]]]
    if task_id in PAIR_LIST_TASKS:
        return [[[]], [[(1, 2), (3, 4)]], [[("a", "b")]]]
    if task_id in {"gcd_two", "lcm_two"}:
        return [[0, 6], [12, 18], [7, 5]]
    if task_id == "clamp_values":
        return [[[], 0, 1], [[-2, 0, 3], 0, 2], [[5, 5], 5, 5]]
    if task_id == "chunks_fixed_size":
        return [[[], 2], [[1, 2, 3, 4, 5], 2], [[1], 3]]
    if task_id in {"merge_sorted_ascending"}:
        return [[[], []], [[1, 3, 5], [2, 4]], [[1, 1], [1, 2]]]
    if task_id in {"merge_sorted_descending"}:
        return [[[], []], [[5, 3, 1], [4, 2]], [[2, 1], [2, 2]]]
    if task_id in {
        "ordered_intersection",
        "ordered_unique_intersection",
        "is_subset",
        "are_disjoint",
    }:
        return [[[], []], [[1, 2, 2, 3], [2, 4]], [[1, 3], [2, 4]]]
    if task_id in {
        "contains_target",
        "linear_first_index",
        "linear_last_index",
        "binary_search_exact",
        "binary_search_insertion",
    }:
        return [[[], 1], [[1, 2, 3, 3, 5], 3], [[1, 2, 4], 3]]
    if task_id in {"count_above", "count_below", "partition_by_threshold"}:
        return [[[], 0], [[-2, 0, 3, 5], 1], [[1, 1], 1]]
    if task_id == "gcd_list":
        return [[[0]], [[12, 18, 24]], [[7, 14]]]
    if task_id == "lcm_list":
        return [[[1]], [[2, 3, 4]], [[5, 10]]]
    if task_id in {"scan_max", "scan_min", "most_frequent", "least_frequent"}:
        return [[[1]], [[3, -1, 3, 2]], [[0, 0, 1]]]
    if task_id in {"rotate_left_one", "rotate_right_one", "minmax_normalize"}:
        return [[[1]], [[1, 2, 3]], [[-2, 0, 5]]]
    return [[[]], [[-3, -1, 0, 2, 2, 5]], [[1, 2, 3, 4]]]


SAFE_BUILTINS = {
    "abs": abs,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "str": str,
}


def _call(source: str, function_name: str, case: list[Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec(compile(source, "<differential>", "exec"), namespace, namespace)
    function = namespace[function_name]
    arguments = copy.deepcopy(case)
    try:
        value = function(*arguments)
        return {
            "outcome": "return",
            "value": value,
            "value_repr": repr(value),
            "mutated_arguments": arguments,
            "mutated_repr": repr(arguments),
        }
    except Exception as error:  # compared by type, as preregistered
        return {
            "outcome": "exception",
            "exception_type": type(error).__name__,
            "mutated_arguments": arguments,
            "mutated_repr": repr(arguments),
        }


def _equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["outcome"] != right["outcome"]:
        return False
    if left["outcome"] == "exception":
        if left["exception_type"] != right["exception_type"]:
            return False
    elif left["value"] != right["value"]:
        return False
    return left["mutated_arguments"] == right["mutated_arguments"]


def execute_pair(payload: dict[str, Any]) -> dict[str, Any]:
    cases = deterministic_cases(payload["task_id"])
    results = []
    passed = True
    for case in cases:
        canonical = _call(
            payload["canonical_source"], payload["canonical_function_name"], case
        )
        opaque = _call(payload["opaque_source"], payload["opaque_function_name"], case)
        equal = _equivalent(canonical, opaque)
        passed = passed and equal
        results.append(
            {
                "input_repr": repr(case),
                "passed": equal,
                "canonical": {
                    key: value
                    for key, value in canonical.items()
                    if key not in {"value", "mutated_arguments"}
                },
                "opaque": {
                    key: value
                    for key, value in opaque.items()
                    if key not in {"value", "mutated_arguments"}
                },
            }
        )
    return {
        "program_id": payload["program_id"],
        "task_id": payload["task_id"],
        "passed": passed,
        "case_count": len(cases),
        "cases": results,
    }


def run_differential(
    payloads: list[dict[str, Any]],
    *,
    max_workers: int = 4,
    timeout_seconds: int = 180,
) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_pair, payload): payload for payload in payloads}
        done, pending = wait(futures, timeout=timeout_seconds)
        results = [future.result() for future in done]
        for future in pending:
            future.cancel()
            payload = futures[future]
            results.append(
                {
                    "program_id": payload["program_id"],
                    "task_id": payload["task_id"],
                    "passed": False,
                    "case_count": 0,
                    "error": "differential execution timeout",
                }
            )
    return sorted(results, key=lambda row: row["program_id"])
