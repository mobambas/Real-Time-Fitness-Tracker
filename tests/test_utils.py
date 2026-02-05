import ast
from pathlib import Path

import math


def load_functions_from_file(file_path, function_names):
    source = Path(file_path).read_text()
    module = ast.parse(source)
    class NPProxy:
        pi = math.pi

        @staticmethod
        def array(values):
            return values

        @staticmethod
        def arctan2(y, x):
            return math.atan2(y, x)

        @staticmethod
        def abs(value):
            return abs(value)

    namespace = {"np": NPProxy}
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            func_source = ast.get_source_segment(source, node)
            exec(func_source, namespace)
    return {name: namespace[name] for name in function_names}


def test_calculate_angle_basic_cases():
    functions = load_functions_from_file("bicep_curls.py", {"calculate_angle"})
    calculate_angle = functions["calculate_angle"]

    angle_right = calculate_angle((1, 0), (0, 0), (0, 1))
    assert round(angle_right) == 90

    angle_straight = calculate_angle((-1, 0), (0, 0), (1, 0))
    assert round(angle_straight) == 180


def test_format_time_zero_and_minutes():
    functions = load_functions_from_file("bicep_curls.py", {"format_time"})
    format_time = functions["format_time"]

    assert format_time(0) == "00:00"
    assert format_time(59) == "00:59"
    assert format_time(60) == "01:00"
    assert format_time(61) == "01:01"
