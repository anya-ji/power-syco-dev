#!/usr/bin/env python3
"""Build the dashboard (if needed) and serve it on localhost. One command.

    uv run python scripts/analysis/serve_dashboard.py

Prints a http://localhost:PORT URL. On a remote box, VS Code Remote and the
JetBrains/Cursor equivalents auto-forward the port, so the URL opens directly on
your laptop; otherwise the printed `ssh -L` line sets up the tunnel by hand.

Binds to 127.0.0.1 by default: this is a shared machine, and the run directory
contains full model outputs. --host 0.0.0.0 exposes it to the whole network.

To publish it instead, leave the bind at 127.0.0.1 and point a tunnel at the
port -- scripts/analysis/tunnel.sh does that. Two flags exist for it:
--build-only rebuilds the pages in place without serving (how a running tunnel
picks up new data), and --auth puts a password in front of a server whose port
is no longer private.
"""

import argparse
import base64
import contextlib
import functools
import hmac
import http.server
import socket
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.artifacts import RunPaths, latest_run
from sycophancy.config import (
    DEFAULT_EXPERIMENT, DEFAULT_RESULTS_DIR, EXPERIMENTS, ROOT as ROOT_DIR,
    results_dir,
)
from sycophancy.dashboard import build, build_index


class Server(socketserver.ThreadingTCPServer):
    """Threaded so one slow download does not block the page's other requests."""

    allow_reuse_address = True
    daemon_threads = True


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the run directory, defaulting to the dashboard, without log spam."""

    def log_message(self, fmt, *args):  # noqa: A003
        pass

    #: page served for "/" -- the run dashboard, or the multi-experiment index
    default_page = "/dashboard.html"

    #: expected Authorization header, or None to serve without a password
    auth_header: str | None = None

    def authorized(self) -> bool:
        """True when no password is set, or the request carries the right one.

        Answers the 401 itself when it does not, so callers can just return.
        The comparison is constant-time: once a tunnel is up this handler is
        reachable by anyone, and a plain == leaks the password a byte at a time.
        """
        if self.auth_header is None:
            return True
        given = self.headers.get("Authorization", "")
        if hmac.compare_digest(given, self.auth_header):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="sycophancy dashboard"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_HEAD(self):  # noqa: N802
        if not self.authorized():
            return None
        return super().do_HEAD()

    def do_GET(self):  # noqa: N802
        if not self.authorized():
            return None
        if self.path == "/":
            self.path = self.default_page
        try:
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            # Browsers routinely abort mid-download (reload, navigate away).
            # The dashboard is tens of MB, so this is normal, not an error.
            return None

    def end_headers(self):
        # The file is rebuilt in place; never let a stale copy be cached.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def free_port(preferred: int, host: str) -> int:
    """Use ``preferred`` if it is free, otherwise let the OS pick one.

    SO_REUSEADDR mirrors what the server itself sets; without it a socket left
    in TIME_WAIT from a previous run makes the preferred port look occupied and
    the URL changes on every restart.
    """
    for candidate in (preferred, 0):
        with contextlib.closing(socket.socket()) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise SystemExit(f"could not bind {host}:{preferred} or any free port")


def handler_for(root: Path, default_page: str, auth: str | None):
    """Handler class bound to one root, default page and optional password."""
    attrs: dict[str, object] = {"default_page": default_page}
    if auth:
        token = base64.b64encode(auth.encode()).decode()
        attrs["auth_header"] = f"Basic {token}"
    handler = type("Handler", (QuietHandler,), attrs)
    return functools.partial(handler, directory=str(root))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=EXPERIMENTS,
                    help="which experiment directory to work in")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override; defaults to <experiment>/results")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory (default: most recent under --results-dir)")
    ap.add_argument("--port", type=int, default=8000,
                    help="preferred port; a free one is chosen if taken")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: localhost only)")
    ap.add_argument("--max-chars", type=int, default=800,
                    help="preview length per text field (default: 800)")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="cap rows embedded for the explorer, stratified by "
                         "model x condition; 0 = embed all. Aggregates always "
                         "use every row. Matches build_dashboard.py -- without "
                         "it a 74k-row run builds a 137 MB page that barely "
                         "opens.")
    ap.add_argument("--no-build", action="store_true",
                    help="serve the existing dashboard.html without rebuilding")
    ap.add_argument("--build-only", action="store_true",
                    help="rebuild every experiment's pages and the entry page, "
                         "then exit without serving. Pages are plain files, so "
                         "this is how a server or tunnel that is already running "
                         "picks up a new run.")
    ap.add_argument("--auth", default=None, metavar="USER:PASS",
                    help="require HTTP basic auth. Worth setting whenever the "
                         "port is reachable beyond localhost -- the pages carry "
                         "full model outputs.")
    ap.add_argument("--pin", action="append", default=[], metavar="EXP=RUN",
                    help="in --all mode, force an experiment to a specific run "
                         "directory instead of its most recent one, e.g. "
                         "--pin exp2=exp2/results/exp2_245x101. Repeatable. "
                         "Without this the newest judged run wins, so finishing "
                         "a new run silently replaces what the dashboard shows.")
    ap.add_argument("--all", action="store_true",
                    help="serve every experiment behind an entry page instead of "
                         "one run")
    ap.add_argument("--open", action="store_true",
                    help="also try to open a local browser")
    return ap.parse_args()


def serve(root: Path, args, extra: list[str], default_page: str = "/dashboard.html") -> None:
    """Serve ``root`` and print where to find things."""
    port = free_port(args.port, args.host)
    handler = handler_for(root, default_page, getattr(args, "auth", None))
    with Server((args.host, port), handler) as httpd:
        url = f"http://localhost:{port}/"
        say = functools.partial(print, flush=True)
        say()
        say("=" * 64)
        say(f"  Open:  {url}")
        for line in extra:
            say(f"         {url}{line}")
        say(f"  Root:  {root}")
        say("=" * 64)
        if args.host == "127.0.0.1":
            say("  Remote? VS Code forwards this automatically. Otherwise run")
            say(f"  on your laptop:  ssh -N -L {port}:localhost:{port} "
                f"{socket.gethostname()}")
        say("  Ctrl-C to stop.")
        say()
        if args.open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            say("\nstopped")


def input_pages(root: Path) -> dict[str, tuple[Path, list[Path], object]]:
    """Each experiment's run-independent page: what goes *into* the run.

    Roles, composed prompts, stimuli, judge rubrics -- none of which need a
    finished run, which is exactly when someone wants to look at them. Listed
    as (destination, source files, builder) so ``build_input_pages`` can skip a
    page whose sources are missing and rebuild one whose sources moved.
    """
    from sycophancy import roles_dashboard, salad_dashboard
    from sycophancy.config import (
        DOMAIN_STATUSES, GENERIC_STATUSES, MODEL_STATUSES,
        SALAD_DOMAIN_STATUSES, SALAD_MODEL_STATUSES, SALAD_SAMPLE,
    )

    return {
        "exp2": (root / "exp2" / "dashboard" / "roles.html",
                 [GENERIC_STATUSES, DOMAIN_STATUSES, MODEL_STATUSES],
                 roles_dashboard.build),
        "exp3": (root / "exp3" / "dashboard" / "inputs.html",
                 [SALAD_SAMPLE, SALAD_DOMAIN_STATUSES, SALAD_MODEL_STATUSES,
                  GENERIC_STATUSES],
                 salad_dashboard.build),
    }


def all_judged(results: Path) -> list[Path]:
    """Every ``<run>/raw/judged.jsonl`` under a results directory, oldest first.

    Judged output lives one level deeper than the run root, so a ``*/judged.jsonl``
    glob matches nothing here.

    Oldest first because the order becomes the page's run tabs, and an
    experiment's runs read chronologically: the design it started with, then
    what was tried next.
    """
    if not Path(results).exists():
        return []
    return sorted(Path(results).glob("*/raw/judged.jsonl"),
                  key=lambda p: p.stat().st_mtime)


def latest_judged(results: Path) -> list[Path]:
    """Just the most recent run's judged output."""
    return all_judged(results)[-1:]


def _stale(dest: Path, sources: list[Path], builder=None) -> bool:
    """True if ``dest`` is missing, or older than its data or its renderer.

    The renderer counts: a template fix that reaches no already-built page is
    the same bug as stale data, and harder to notice.
    """
    if not dest.exists():
        return True
    files = list(sources)
    if builder is not None:
        module = getattr(builder, "__module__", None)
        mod = sys.modules.get(module) if module else None
        if mod is not None and getattr(mod, "__file__", None):
            files.append(Path(mod.__file__))
    newest = max(f.stat().st_mtime for f in files if Path(f).exists())
    return dest.stat().st_mtime < newest


def build_input_pages(root: Path, only: str | None = None,
                      force: bool = False) -> dict[str, Path]:
    """Build the input pages that are out of date; return every one that exists.

    Skipped when the sources are not there yet -- exp3's page needs
    ``scripts/data/sample_salad.py`` to have run -- so a fresh checkout serves
    what it can rather than failing on what it cannot.
    """
    out: dict[str, Path] = {}
    for exp, (dest, sources, builder) in input_pages(root).items():
        if only and exp != only:
            continue
        if not all(Path(s).exists() for s in sources):
            continue
        if force or _stale(dest, sources, builder):
            builder(dest)
        out[exp] = dest
    return out


def resolve_pins(pins: list[str], root: Path) -> dict[str, Path]:
    """Parse ``EXP=RUN`` overrides into run directories, checking each exists."""
    out = {}
    for pin in pins:
        exp, _, run = pin.partition("=")
        if not run:
            raise SystemExit(f"--pin needs EXP=RUN, got {pin!r}")
        path = Path(run)
        if not path.is_absolute():
            path = root / path
        # Accept either the run root or its judged file, since both are things
        # a caller is likely to have on hand.
        if path.name == "judged.jsonl":
            path = path.parent.parent
        if not (path / "raw" / "judged.jsonl").exists():
            raise SystemExit(f"--pin {exp}: no raw/judged.jsonl under {path}")
        out[exp] = path
    return out


def build_all(args) -> list[str]:
    """Build every experiment's pages and the entry page; return their links.

    Separate from serving so a dashboard that is already published can be
    refreshed: the pages are plain files, so rewriting them here is picked up by
    a running server and tunnel on the next request, with nothing restarted and
    no URL changing.
    """
    from sycophancy.config import ROOT

    pages = {}
    pins = resolve_pins(getattr(args, "pin", []), ROOT)
    if not args.no_build:
        for exp in sorted(d for d in ROOT.iterdir()
                          if d.is_dir() and d.name.startswith("exp")):
            pinned = pins.get(exp.name)
            # Every judged run becomes a tab on the experiment's page, so a new
            # design sits beside the one it came from instead of replacing it.
            # A pin still means one run: it exists to look at that run alone.
            judged = ([pinned / "raw" / "judged.jsonl"] if pinned
                      else all_judged(exp / "results"))
            if not judged:
                continue
            runs = [RunPaths(j.parent.parent) for j in judged]
            dest = exp / "dashboard" / "dashboard.html"
            dest.parent.mkdir(parents=True, exist_ok=True)
            # These pages run to tens of MB; rebuild only when a run they
            # summarise is newer than the page. A pin has to override that:
            # the page it replaces is usually *newer* than the run pinned to.
            if pinned or _stale(dest, judged, build):
                print(f"  {exp.name}: {', '.join(r.root.name for r in runs)}"
                      f"{'  (pinned)' if pinned else ''}")
                build(runs, dest, max_chars=args.max_chars,
                      max_rows=None if args.max_rows == 0 else args.max_rows)
        # The stimulus/roles pages exist before any run does, so they are built
        # here too -- one command has to be enough to see everything.
        pages = build_input_pages(ROOT)
    build_index(ROOT)

    links = [str(p.relative_to(ROOT))
             for p in sorted(ROOT.glob("exp*/dashboard/dashboard.html"))]
    links += [str(p.relative_to(ROOT)) for p in pages.values()]
    return links


def serve_all(args) -> None:
    """Entry page across every experiment."""
    from sycophancy.config import ROOT

    serve(ROOT, args, build_all(args), default_page="/index.html")


def main() -> None:
    args = parse_args()
    if args.build_only:
        print("Rebuilt:")
        for link in build_all(args):
            print(f"  {link}")
        return
    if args.all:
        return serve_all(args)

    args.results_dir = args.results_dir or results_dir(args.experiment)

    # No run yet? Serve what the experiment does have -- its stimuli, roles and
    # conditions -- instead of failing. One command works either way.
    try:
        paths = RunPaths(args.run) if args.run else latest_run(args.results_dir)
    except FileNotFoundError:
        pages = build_input_pages(
            ROOT_DIR, only=args.experiment, force=not args.no_build
        )
        page = pages.get(args.experiment)
        if page is None:
            raise SystemExit(
                f"{args.experiment} has no run under {args.results_dir} and no "
                f"inputs page to fall back to; run its pipeline first"
            ) from None
        print(f"No run under {args.results_dir} yet — serving the inputs page")
        return serve(page.parent, args, [], default_page=f"/{page.name}")

    if args.no_build and paths.dashboard.exists():
        print(f"Serving existing {paths.dashboard.name}")
    else:
        build(paths, max_chars=args.max_chars,
              max_rows=None if args.max_rows == 0 else args.max_rows)

    port = free_port(args.port, args.host)
    handler = handler_for(paths.root, "/dashboard.html", args.auth)

    with Server((args.host, port), handler) as httpd:
        url = f"http://localhost:{port}/"
        say = functools.partial(print, flush=True)
        say()
        say("=" * 64)
        say(f"  Dashboard:  {url}")
        say(f"  Run:        {paths.root}")
        say(f"  Also here:  {url}report/report.pdf")
        say(f"              {url}figures/")
        say("=" * 64)
        if args.host == "127.0.0.1":
            host = socket.gethostname()
            say(f"  Remote? VS Code forwards this automatically. Otherwise run")
            say(f"  on your laptop:  ssh -N -L {port}:localhost:{port} {host}")
        say("  Ctrl-C to stop.")
        say()
        if args.open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
