# Import other utilities as needed
from .io_utils import load_features, export_results

# Lazy import to avoid pyopenms dependency issues

def build_roi(paths, features, plot, args):
    from .extract_eic import build
    return build(paths, features, plot, args)