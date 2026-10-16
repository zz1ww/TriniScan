"""最小 pytest 兼容层：仅支持本仓库测试用到的 assert/raises/approx。
用法: python tools/_mini_pytest.py tests/test_xxx.py [更多...]
"""
import sys, os, types, traceback, importlib.util

def _make_pytest():
    m = types.ModuleType("pytest")
    class _Raises:
        def __init__(self, exc, match=None): self.exc=exc; self.match=match
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            if et is None: raise AssertionError(f"未抛出 {self.exc}")
            if not issubclass(et, self.exc): return False
            if self.match:
                import re
                if not re.search(self.match, str(ev)): raise AssertionError(
                    f"异常信息不匹配 {self.match!r}: {ev}")
            return True
    class _Approx:
        def __init__(self, v, rel=None, abs=None): self.v=v; self.rel=rel; self.abs=abs
        def __eq__(self, o):
            import math
            a,b=float(self.v),float(o)
            abs_t = self.abs if self.abs is not None else 1e-9
            rel_t = self.rel if self.rel is not None else 1e-6
            return math.isclose(a,b,rel_tol=rel_t,abs_tol=abs_t)
        def __repr__(self): return f"approx({self.v})"
    m.raises=_Raises; m.approx=_Approx
    def _importorskip(name, *a, **k):
        import importlib
        try:
            return importlib.import_module(name)
        except ImportError:
            raise KeyboardInterrupt(f"skip: {name} 未安装")
    m.importorskip=_importorskip
    def _mark_stub(*a, **k):
        def deco(f): return f
        return f if (len(a)==1 and callable(a[0])) else deco
    m.mark=types.SimpleNamespace(parametrize=_mark_stub, skip=_mark_stub,
                                 skipif=_mark_stub, xfail=_mark_stub)
    m.skip=lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt("skip"))
    fixtures={}
    def _fixture(*a, **k):
        def deco(f):
            fixtures[f.__name__]=f
            return f
        # 支持 @pytest.fixture 或 @pytest.fixture() 两种写法
        if len(a)==1 and callable(a[0]) and not k:
            return deco(a[0])
        return deco
    m.fixture=_fixture
    m._fixtures=fixtures
    return m

def run(path):
    sys.path.insert(0, os.getcwd())
    spec=importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod=importlib.util.module_from_spec(spec)
    pt=_make_pytest()
    sys.modules["pytest"]=pt
    spec.loader.exec_module(mod)
    fixtures=dict(pt._fixtures)
    n=ok=fail=0
    import inspect, tempfile, shutil, pathlib, itertools
    for name in dir(mod):
        if not name.startswith("test_") or not callable(getattr(mod,name)):
            continue
        n+=1
        fn=getattr(mod,name); kw={}; fin=[]; tmpdir=None
        for pnm in inspect.signature(fn).parameters:
            if pnm=="tmp_path":
                tmpdir=tempfile.mkdtemp(); kw[pnm]=pathlib.Path(tmpdir)
            elif pnm in fixtures:
                val=fixtures[pnm]()
                if inspect.isgenerator(val):
                    gen=val; val=next(gen); fin.append(gen)
                kw[pnm]=val
        try:
            fn(**kw); ok+=1; print(f"  PASS {name}")
        except KeyboardInterrupt: print(f"  SKIP {name}")
        except Exception:
            fail+=1; print(f"  FAIL {name}"); traceback.print_exc()
        finally:
            for g in fin:
                try: next(g)
                except StopIteration: pass
            if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"{os.path.basename(path)}: {ok}/{n} passed, {fail} failed")
    return fail

if __name__=="__main__":
    import glob
    paths=sys.argv[1:] or sorted(glob.glob("tests/test_*.py"))
    total=sum(run(p) for p in paths)
    sys.exit(1 if total else 0)
