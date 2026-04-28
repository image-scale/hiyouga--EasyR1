import re


def parse_log(log: str) -> dict[str, str]:
    """Parse test runner output into per-test results.

    Args:
        log: Full stdout+stderr output of `bash run_test.sh 2>&1`.

    Returns:
        Dict mapping test_id to status.
        - test_id: pytest native format (e.g. "tests/foo.py::TestClass::test_func[param]")
        - status: one of "PASSED", "FAILED", "SKIPPED", "ERROR"
    """
    results = {}

    # Match pytest verbose output lines like:
    # tests/test_checkpoint.py::test_find_latest_ckpt PASSED                   [  6%]
    # tests/test_dataproto.py::test_repeat[True] PASSED                        [ 50%]
    test_line_re = re.compile(
        r"^(\S+::.*?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s*(?:\[.*\])?\s*$"
    )

    for line in log.splitlines():
        line = line.strip()
        m = test_line_re.match(line)
        if m:
            test_id = m.group(1)
            status = m.group(2)
            # Normalize xfail/xpass
            if status == "XFAIL":
                status = "SKIPPED"
            elif status == "XPASS":
                status = "PASSED"
            results[test_id] = status

    # Handle collection errors: lines like "ERROR tests/foo.py" (no ::)
    collection_error_re = re.compile(r"^ERROR\s+(tests/\S+\.py)\s*$")
    for line in log.splitlines():
        line = line.strip()
        m = collection_error_re.match(line)
        if m:
            module_id = m.group(1)
            results[module_id] = "ERROR"

    return results

