from lite.methods.names import normalize_md22_method_name
from lite.plotting.io import read_results as _read_results


def read_results(path):
    frame = _read_results(path)
    if "method" in frame:
        frame["method"] = frame["method"].map(
            lambda name: (
                "lite_normalized"
                if str(name).lower() == "lite_normalized"
                else normalize_md22_method_name(name)
            )
        )
    return frame
