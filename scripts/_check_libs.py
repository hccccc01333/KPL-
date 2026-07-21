import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for lib in ['xgboost', 'lightgbm', 'catboost', 'sklearn']:
    try:
        m = __import__(lib)
        ver = getattr(m, '__version__', 'unknown')
        print(f"  {lib:<12} ✅ {ver}")
    except ImportError:
        print(f"  {lib:<12} ❌ MISSING")
