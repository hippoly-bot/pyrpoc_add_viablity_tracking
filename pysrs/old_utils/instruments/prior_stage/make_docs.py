# inspect the DLL because they dont give me docs

from ctypes import WinDLL
import inspect

# Path to the DLL (Modify this if necessary)
DLL_PATH = r"C:\Users\Lab Admin\Documents\PythonStuff\pysrs\pysrs\instruments\prior_stage\PriorScientificSDK.dll"

# Load the DLL
try:
    SDKPrior = WinDLL(DLL_PATH)
    print("[✓] DLL Loaded Successfully!")
except Exception as e:
    print(f"[X] Failed to load DLL: {e}")
    exit()

# List exported functions
print("\n[🔍] Exported Functions in DLL:")
functions = [func for func in dir(SDKPrior) if not func.startswith("_")]

if not functions:
    print("[X] No functions found! DLL may not export symbols.")
else:
    for func in functions:
        print(" -", func)

# Try to get function signatures if possible
print("\n[🔍] Function Details (If Available):")
for func in functions:
    try:
        function_obj = getattr(SDKPrior, func)
        sig = inspect.signature(function_obj)
        print(f" - {func}{sig}")
    except ValueError:
        print(f" - {func} (No signature available)")

import ctypes

DLL_PATH = r"C:\Users\Lab Admin\Documents\PythonStuff\pysrs\pysrs\instruments\prior_stage\PriorScientificSDK.dll"

try:
    SDKPrior = ctypes.windll.LoadLibrary(DLL_PATH)
    print("[✓] DLL Loaded Successfully!")
except Exception as e:
    print(f"[X] Failed to load DLL: {e}")
