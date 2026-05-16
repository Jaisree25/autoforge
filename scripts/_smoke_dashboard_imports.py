"""Replicate what `streamlit run dashboard/app.py` does to imports.

Streamlit sets sys.path[0] to the script's directory (dashboard/), NOT the
project root. We mimic that here so we catch any "from config import ..."-style
breakage WITHOUT needing to spin up a real Streamlit server + browser.

The dashboard app.py self-prepends the project root early — this script proves
that fix is sufficient.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
APP_PATH = DASHBOARD_DIR / "app.py"

# Reset sys.path to what `streamlit run` would give us.
sys.path = [str(DASHBOARD_DIR)] + [p for p in sys.path if p not in (".", str(PROJECT_ROOT))]


def _stub_streamlit() -> None:
    """Make `import streamlit` and streamlit_autorefresh no-op so the script can
    run module-level code outside a real Streamlit runtime."""
    import types

    if "streamlit" in sys.modules:
        return

    # Subclass ModuleType so unknown attributes fall through to a passthrough.
    # This keeps the stub maintenance-free as components add new st.* calls.
    class _StubModule(types.ModuleType):
        def __getattr__(self, name):  # only called when name isn't set
            return _AnyAttr()

    fake = _StubModule("streamlit")

    class _AnyAttr:
        def __getattr__(self, _name): return _AnyAttr()
        def __call__(self, *args, **kwargs):
            # Behave like a selectbox/multiselect/radio: return the first option.
            if "options" in kwargs:
                opts = kwargs["options"]
                if opts:
                    return next(iter(opts))
            for a in args:
                if isinstance(a, (list, tuple, dict)) and a:
                    return next(iter(a))
            return _AnyAttr()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __bool__(self): return False
        def __iter__(self): return iter(())

    def _stop():
        raise SystemExit(0)  # mimic st.stop

    def _passthrough(*args, **kwargs):
        # st.sidebar.selectbox needs to return something; first option is fine
        if "options" in kwargs:
            opts = kwargs["options"]
            if opts:
                return next(iter(opts))
        if len(args) >= 2 and isinstance(args[1], (list, dict)):
            opts = args[1]
            if opts:
                return next(iter(opts))
        return _AnyAttr()

    fake.set_page_config = lambda **kw: None
    fake.cache_resource = lambda fn=None, **kw: (fn if fn else (lambda f: f))
    fake.session_state = {}
    fake.stop = _stop
    fake.sidebar = _AnyAttr()
    fake.title = _passthrough
    fake.subheader = _passthrough
    fake.header = _passthrough
    fake.caption = _passthrough
    fake.markdown = _passthrough
    fake.warning = _passthrough
    fake.info = _passthrough
    fake.error = _passthrough
    fake.code = _passthrough
    fake.divider = _passthrough
    fake.button = _passthrough
    fake.selectbox = _passthrough
    fake.text_area = _passthrough
    fake.columns = lambda spec, **kw: [_AnyAttr() for _ in (spec if isinstance(spec, (list, tuple)) else range(spec))]
    fake.container = _passthrough
    fake.expander = _passthrough
    fake.json = _passthrough
    fake.rerun = _passthrough
    sys.modules["streamlit"] = fake

    autorefresh = _StubModule("streamlit_autorefresh")
    autorefresh.st_autorefresh = lambda **kw: None
    sys.modules["streamlit_autorefresh"] = autorefresh

    # streamlit.components.v1 submodule — used by components.html() iframe pattern.
    sys.modules["streamlit.components"] = _StubModule("streamlit.components")
    sys.modules["streamlit.components.v1"] = _StubModule("streamlit.components.v1")

    # streamlit-agraph + streamlit-extras land in wave 6 / dev iteration; stub
    # them too so this smoke script doesn't grow a dependency over time.
    for mod_name in ("streamlit_agraph", "streamlit_extras"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _StubModule(mod_name)


def main() -> int:
    _stub_streamlit()
    try:
        runpy.run_path(str(APP_PATH), run_name="__main__")
    except SystemExit as e:
        # st.stop() raises SystemExit(0) — that's a clean exit
        code = e.code if isinstance(e.code, int) else 0
        if code == 0:
            print("dashboard module imports + executes cleanly (st.stop hit)")
            return 0
        print(f"unexpected SystemExit({code})")
        return code
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print("dashboard module imports + executes cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
