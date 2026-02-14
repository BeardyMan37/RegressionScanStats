import fnmatch
from pathlib import Path
import re
from typing import Iterable, NamedTuple


class QA2FlagMapping(NamedTuple):
    # Pattern to match. Exact match if string, otherwise, regex pattern.
    pattern: str | re.Pattern[str]
    value: int


def parse_qa2flag_mapping(lines: Iterable[str]):
    qa2flag_mapping_list: list[QA2FlagMapping] = list()
    valid_pattern = re.compile(
        "((?P<pattern_type>wildcard|regex):)?(?P<pattern>.+?) +-> +(?P<value>[+-]?[0-9]+)"
    )
    for line in lines:
        match_result = valid_pattern.match(line)
        if match_result is None:
            raise ValueError(f"Invalid qa2flag mapping found: {line}")
        pattern = match_result.group("pattern")
        pattern_type = match_result.group("pattern_type")
        value = int(match_result.group("value"))
        if pattern_type == "wildcard":
            pattern = fnmatch.translate(pattern)
            pattern_type = "regex"
        if pattern_type == "regex":
            pattern = re.compile(pattern)
        qa2flag_mapping_list.append(QA2FlagMapping(pattern=pattern, value=value))
    return qa2flag_mapping_list


def parse_qa2flag_mapping_file(path: Path | str):
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return parse_qa2flag_mapping(open(path, "r"))


def transform(
    qa2flags: Iterable[str], qa2flag_mapping_list: list[QA2FlagMapping], default: int
):
    for qa2flag_mapping in qa2flag_mapping_list:
        if isinstance(qa2flag_mapping.pattern, str):
            for qa2flag in qa2flags:
                if qa2flag == qa2flag_mapping.pattern:
                    return qa2flag_mapping.value
        else:
            for qa2flag in qa2flags:
                if qa2flag_mapping.pattern.match(str(qa2flag)) is not None:
                    return qa2flag_mapping.value
    return default
