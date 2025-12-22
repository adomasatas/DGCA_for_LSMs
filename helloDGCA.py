import numpy as np
import argparse
import shutil

# Prefer graph-tool (if available), else fallback to networkx
HAVE_GT = False
try:
    import graph_tool.all as gt
    HAVE_GT = True
except Exception:
    HAVE_GT = False

# no NetworkX fallback — we rely on graph-tool for drawing

from grow.dgca import DGCA
from grow.reservoir import Reservoir
import os
import traceback

# Draw mode: choose interactive window or PNG output via CLI
# When True, monkey-patch GdkPixbuf to skip/replace loading SVG logos that crash
# on some macOS installs. This creates a tiny blank Pixbuf instead of failing.
SKIP_GRAPH_TOOL_LOGO = True


def _safe_filename(title: str) -> str:
    # sanitize title for filenames
    fname = title.replace(' ', '_').replace(':', '').replace('/', '_')
    if fname == '':
        fname = 'step'
    out_dir = 'output'
    # ensure output directory exists (main() will wipe it on non-interactive runs)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"out_{fname}.png")


def draw_reservoir(res: Reservoir, title: str = "", prev_pos=None, savefile: str | None = None, interactive: bool = False):
    """Visualize the reservoir using graph-tool.
    If interactive is False the function writes a PNG to `savefile` and returns a pos dict when available.
    If interactive is True it opens an interactive GTK window (may require GUI support).
    """
    if res is None or res.size() == 0:
        print("[draw] empty reservoir, skipping draw")
        return None

    # determine savefile if not provided
    if savefile is None:
        savefile = _safe_filename(title)

    # Use graph-tool with pretty properties (no networkx fallback)
    if not HAVE_GT:
        print("[draw] graph-tool not available; skipping draw")
        return None

    try:
        g = res.to_gt(pp=True)

        # Optional monkey-patch to avoid SVG logo parsing errors
        _patched = False
        _orig_new_from_file = None
        if SKIP_GRAPH_TOOL_LOGO:
            try:
                import gi
                gi.require_version('GdkPixbuf', '2.0')
                from gi.repository import GdkPixbuf

                _orig_new_from_file = getattr(GdkPixbuf.Pixbuf, 'new_from_file', None)

                def _patched_new_from_file(filename):
                    try:
                        if isinstance(filename, str) and filename.lower().endswith('.svg'):
                            return GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 16, 16)
                    except Exception:
                        pass
                    if _orig_new_from_file is not None:
                        return _orig_new_from_file(filename)
                    raise RuntimeError('no GdkPixbuf loader available')

                if _orig_new_from_file is not None:
                    GdkPixbuf.Pixbuf.new_from_file = _patched_new_from_file
                    _patched = True
            except Exception:
                _patched = False

        try:
            # call graph-tool drawing functions and catch their drawing-specific errors
            if interactive:
                try:
                    gt.interactive_window(
                        g,
                        pos=g.vp.get('pos', None),
                        vertex_fill_color=g.vp.get('plot_color', None),
                        vertex_color=g.vp.get('outline_color', None),
                        edge_color=g.ep.get('edge_color', None) if hasattr(g, 'ep') else None,
                    )
                    print(f"[draw] opened interactive graph-tool window for '{title}'")
                except Exception as ie:
                    print('[draw] interactive graph-tool failed:', ie)
                    if interactive:
                        print('[draw] environment:')
                        print('DISPLAY=', os.environ.get('DISPLAY'))
                        print('WAYLAND_DISPLAY=', os.environ.get('WAYLAND_DISPLAY'))
                        print('GDK_BACKEND=', os.environ.get('GDK_BACKEND'))
                    # return gracefully when interactive drawing fails
                    return None
            else:
                try:
                    gt.graph_draw(
                        g,
                        pos=g.vp.get('pos', None),
                        vertex_fill_color=g.vp.get('plot_color', None),
                        vertex_color=g.vp.get('outline_color', None),
                        edge_color=g.ep.get('edge_color', None) if hasattr(g, 'ep') else None,
                        output=savefile,
                    )
                    print(f"[draw] wrote {savefile} (graph-tool)")
                except Exception as ge:
                    # handle drawing errors (empty graph / degenerate transforms)
                    print('[draw] graph-tool graph_draw failed:', ge)
                    return None

            # convert graph-tool pos property to a simple dict for caller
            pos_prop = None
            try:
                pos_prop = g.vp.get('pos', None)
            except Exception:
                pos_prop = None

            if pos_prop is not None:
                pos = {int(v): (pos_prop[v][0], pos_prop[v][1]) for v in g.vertices()}
            else:
                pos = None
            return pos
        finally:
            # restore original new_from_file if we patched it
            try:
                if SKIP_GRAPH_TOOL_LOGO and _patched and _orig_new_from_file is not None:
                    from gi.repository import GdkPixbuf
                    GdkPixbuf.Pixbuf.new_from_file = _orig_new_from_file
            except Exception:
                pass
    except Exception as e:
        print("[draw] graph-tool draw failed:", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        return None

    # No NetworkX fallback; drawing already handled above via graph-tool


def main():
    parser = argparse.ArgumentParser(description='Draw a small DGCA-grown reservoir.')
    parser.add_argument('--interactive', action='store_true', help='Open an interactive graph-tool window instead of writing PNGs')
    args = parser.parse_args()
    interactive = bool(args.interactive)

    # For non-interactive runs, wipe the output directory so each run is clean
    if not interactive:
        out_dir = 'output'
        if os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
            except Exception:
                pass
        os.makedirs(out_dir, exist_ok=True)

    # np.random.seed(42)  # reproducible MLP weights and NX layouts

    # 3-node chain: Input -> Internal -> Output
    A0 = np.array([
        [0, 0, 1],  # input -> internal
        [0, 0, 0],  # output (no outgoing edges)
        [0, 1, 0]   # internal -> output
    ], dtype=int)

   

    n_states = 3
    hidden_size = 80
    steps = 12

    # all nodes start in the same state (state 0)
    S0 = np.zeros((3, n_states), dtype=int)
    S0[:, 0] = 1

    # define which nodes are input/output
    res = Reservoir(A0, S0, input_nodes=1, output_nodes=1)
    dgca = DGCA(n_states=n_states, hidden_size=hidden_size)

    print(f"Seed: nodes={res.size()}, edges={int(res.A.sum())}")
    prev_pos = None
    prev_pos = draw_reservoir(res, title="Step 0 (seed)", interactive=interactive)

    for t in range(1, steps + 1):
        res = dgca.step(res)
        if res is None or res.size() == 0:
            print(f"Step {t}: produced empty graph, stopping.")
            break
        n_nodes = res.size()
        n_edges = int(res.A.sum())
        print(f"Step {t}: nodes={n_nodes}, edges={n_edges}")
        prev_pos = draw_reservoir(res, title=f"Step {t}: nodes={n_nodes}, edges={n_edges}", prev_pos=prev_pos, interactive=interactive)

if __name__ == "__main__":
    main()