"""最小 pytest 兼容层：让本仓库测试无需安装 pytest 也能跑。

支持本仓库测试用到的特性：
- ``assert`` / ``pytest.raises`` / ``pytest.approx``；
- ``@pytest.fixture``（含递归依赖、生成器 teardown、模块作用域）；
- ``@pytest.mark.parametrize``（单/多参数、笛卡尔积）；
- 内建 fixture ``tmp_path`` / ``monkeypatch``；
- ``pytest.skip`` / ``pytest.importorskip``。

用法::

    python tools/_mini_pytest.py tests/test_xxx.py [更多...]
    python tools/_mini_pytest.py            # 跑 tests/ 下全部
"""
import glob
import importlib.util
import inspect
import itertools
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
import types


# ---------------------------------------------------------------------------
# pytest 兼容 shim
# ---------------------------------------------------------------------------
class _Raises:
    def __init__(self, exc, match=None):
        self.exc = exc
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"未抛出 {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match:
            import re
            if not re.search(self.match, str(ev)):
                raise AssertionError(
                    f"异常信息不匹配 {self.match!r}: {ev}")
        return True


class _Approx:
    def __init__(self, v, rel=None, abs=None):
        self.v, self.rel, self.abs = v, rel, abs

    def __eq__(self, o):
        import math
        a, b = float(self.v), float(o)
        abs_t = self.abs if self.abs is not None else 1e-9
        rel_t = self.rel if self.rel is not None else 1e-6
        return math.isclose(a, b, rel_tol=rel_t, abs_tol=abs_t)

    def __repr__(self):
        return f"approx({self.v})"


class _Skip(Exception):
    """内部跳过信号（对应 pytest.skip）。"""


class _MonkeyPatch:
    """极简 monkeypatch：setattr / delattr / setenv / delenv + 自动还原。"""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str):          # monkeypatch.setattr("a.b.c", v)
            value = name
            target, _, name = target.rpartition(".")
            target = self._resolve(target)
        if not hasattr(target, name) and raising:
            raise AttributeError(f"{target!r} 无属性 {name!r}")
        old = getattr(target, name, None)
        had = hasattr(target, name)
        setattr(target, name, value)
        self._undo.append(("attr", target, name, old, had))

    def delattr(self, target, name, raising=True):
        if isinstance(target, str):
            target, _, name = target.rpartition(".")
            target = self._resolve(target)
        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
            return
        old = getattr(target, name)
        delattr(target, name)
        self._undo.append(("attr", target, name, old, True))

    def setenv(self, name, value):
        import os as _os
        old = _os.environ.get(name)
        _os.environ[name] = str(value)
        self._undo.append(("env", name, old))

    def delenv(self, name, raising=True):
        import os as _os
        if name in _os.environ:
            old = _os.environ[name]
            del _os.environ[name]
            self._undo.append(("env", name, old))
        elif raising:
            raise KeyError(name)

    def undo(self):
        import os as _os
        while self._undo:
            item = self._undo.pop()
            if item[0] == "attr":
                _, target, name, old, had = item
                if had:
                    setattr(target, name, old)
                else:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
            else:
                _, name, old = item
                if old is None:
                    _os.environ.pop(name, None)
                else:
                    _os.environ[name] = old

    @staticmethod
    def _resolve(dotted):
        import importlib
        parts = dotted.split(".")
        for k in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:k]))
            except ImportError:
                continue
            for p in parts[k:]:
                obj = getattr(obj, p)
            return obj
        raise ImportError(dotted)


def _make_pytest():
    m = types.ModuleType("pytest")
    m.raises = _Raises
    m.approx = _Approx
    m.skip = lambda *a, **k: (_ for _ in ()).throw(
        _Skip(a[0] if a else "skip"))

    def _importorskip(name, *a, **k):
        import importlib
        try:
            return importlib.import_module(name)
        except ImportError:
            raise _Skip(f"{name} 未安装")

    m.importorskip = _importorskip

    def _parametrize(argnames, argvalues, **kw):
        if isinstance(argnames, str):
            names = [n.strip() for n in argnames.split(",")]
        else:
            names = list(argnames)

        def deco(fn):
            rows = []
            for val in argvalues:
                vals = val if len(names) > 1 else (val,)
                rows.append(dict(zip(names, vals)))
            cases = getattr(fn, "_params", [])
            fn._params = [dict(a, **b) for a in rows for b in cases] \
                if cases else rows
            return fn
        return deco

    def _id(*a, **k):
        return lambda f: f

    m.mark = types.SimpleNamespace(
        parametrize=_parametrize, skip=_id, skipif=_id, xfail=_id,
        slow=_id)

    fixtures = {}

    def _fixture(*a, **k):
        scope = k.get("scope", "function")

        def deco(f):
            fixtures[f.__name__] = (f, scope)
            return f

        # 支持 @pytest.fixture 与 @pytest.fixture(scope=...) 两种写法
        if len(a) == 1 and callable(a[0]) and not k:
            return deco(a[0])
        return deco

    m.fixture = _fixture
    m._fixtures = fixtures
    m._Skip = _Skip
    m._MonkeyPatch = _MonkeyPatch
    return m


# ---------------------------------------------------------------------------
# fixture 解析
# ---------------------------------------------------------------------------
class _FixturePool:
    """按作用域缓存 fixture 值，递归解析依赖，统一 teardown。"""

    def __init__(self, fixtures, module):
        self._defs = fixtures
        self._module = module
        self._cache = {}
        self._finalizers = []          # [(generator)]

    def get(self, name, tmpdirs, monkeypatch):
        defs = self._defs
        if name not in defs:
            # 内建
            if name == "tmp_path":
                if "_tmp_path" not in self.__dict__:
                    d = pathlib.Path(tempfile.mkdtemp())
                    tmpdirs.append(d)
                    self._tmp_path = d
                return self._tmp_path
            if name == "tmp_path_factory":
                return self._tmp_factory(tmpdirs)
            if name == "monkeypatch":
                return monkeypatch
            raise LookupError(name)
        fn, scope = defs[name]
        if name in self._cache:
            return self._cache[name]
        val = self._call(fn, tmpdirs, monkeypatch)
        if scope in ("module", "session", "package"):
            self._cache[name] = val
        return val

    def _call(self, fn, tmpdirs, monkeypatch):
        kw = {}
        for pnm in inspect.signature(fn).parameters:
            kw[pnm] = self.get(pnm, tmpdirs, monkeypatch)
        val = fn(**kw)
        if inspect.isgenerator(val):
            gen = val
            val = next(gen)
            self._finalizers.append(gen)
        return val

    def _tmp_factory(self, tmpdirs):
        class _F:
            @staticmethod
            def mktemp(name=None):
                d = pathlib.Path(tempfile.mkdtemp())
                tmpdirs.append(d)
                return d if name is None else d / name
        return _F()

    def teardown(self):
        while self._finalizers:
            g = self._finalizers.pop()
            try:
                next(g)
            except StopIteration:
                pass


# ---------------------------------------------------------------------------
def _iter_cases(fn):
    """展开 parametrize；返回 [ (id_suffix, kwargs) ]。"""
    params = getattr(fn, "_params", None)
    if not params:
        return [("", {})]
    out = []
    for i, kw in enumerate(params):
        label = ",".join(f"{k}={v!r}" for k, v in kw.items())
        out.append((f"[{i}:{label}]", kw))
    return out


def run(path):
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(
        os.path.basename(path)[:-3], path)
    if spec is None or spec.loader is None:
        print(f"{os.path.basename(path)}: 无法加载")
        return 1
    mod = importlib.util.module_from_spec(spec)
    pt = _make_pytest()
    sys.modules["pytest"] = pt
    spec.loader.exec_module(mod)

    n = ok = fail = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_") or not callable(getattr(mod, name)):
            continue
        fn = getattr(mod, name)
        for suffix, extra in _iter_cases(fn):
            n += 1
            kw = dict(extra)
            tmpdirs = []
            mp = _MonkeyPatch()
            pool = _FixturePool(dict(pt._fixtures), mod)
            try:
                for pnm in inspect.signature(fn).parameters:
                    if pnm in extra:
                        continue
                    kw[pnm] = pool.get(pnm, tmpdirs, mp)
                try:
                    fn(**kw)
                    ok += 1
                    print(f"  PASS {name}{suffix}")
                except _Skip as exc:
                    print(f"  SKIP {name}{suffix} ({exc})")
                    n -= 1
            except _Skip as exc:
                print(f"  SKIP {name}{suffix} ({exc})")
                n -= 1
            except Exception:
                fail += 1
                print(f"  FAIL {name}{suffix}")
                traceback.print_exc()
            finally:
                try:
                    pool.teardown()
                except Exception:
                    pass
                mp.undo()
                for d in tmpdirs:
                    shutil.rmtree(d, ignore_errors=True)

    print(f"{os.path.basename(path)}: {ok}/{n} passed, {fail} failed")
    return fail


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("tests/test_*.py"))
    total = 0
    for p in paths:
        total += run(p)
    print(f"\n合计: {total} 个失败")
    sys.exit(1 if total else 0)
