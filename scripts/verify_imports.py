import sys, importlib, traceback
from pathlib import Path

def show(msg):
    print("="*60)
    print(msg)
    print("-"*60)

show("python executable")
print(sys.executable)

show("sys.path")
for p in sys.path:
    print(p)

show("pip package info (programmatic)")
try:
    import importlib.metadata as md
    for pkg in ("pandas","scikit-learn","joblib"):
        try:
            dist = md.distribution(pkg)
            print(pkg, "->", dist.version, dist.locate_file(""))
        except Exception as e:
            print(pkg, "NOT FOUND:", e)
except Exception:
    print("importlib.metadata not available")

show("attempt imports")
for mod in ("pandas","sklearn.ensemble","joblib"):
    try:
        __import__(mod)
        print(f"OK: imported {mod}")
    except Exception:
        print(f"ERROR importing {mod}")
        traceback.print_exc()

# Optionally attempt to import IsolationForest directly
show("IsolationForest import test")
try:
    from sklearn.ensemble import IsolationForest
    print("OK: IsolationForest available")
except Exception:
    print("ERROR: IsolationForest import failed")
    traceback.print_exc()