@echo off
setlocal
REM ================================================================
REM Build Python SDK (Windows + OpenVINO backend)
REM
REM Prerequisites:
REM   1. VS2022 BuildTools (C++ desktop workload) or full VS2022
REM   2. CMake >= 3.21
REM   3. Python 3.8+ with openvino installed for runtime detection:
REM        python -m pip install openvino
REM
REM Output: python/fastyolo/libs/ (collected by collect_artifacts.py)
REM ================================================================

set ROOT=%~dp0..\..
cd /d "%ROOT%" || exit /b 1

REM ---- 0. Ensure cmake is on PATH (pip-installed cmake may be in user Scripts) ----
where cmake >nul 2>nul || (
    for /f "delims=" %%i in ('python -c "import site,sys;print(site.USER_BASE + r'"'"'\Python'"'"' + str(sys.version_info[0]) + str(sys.version_info[1]) + r'"'"'\Scripts'"'"')"') do set "PATH=%%i;%PATH%"
)
where cmake >nul 2>nul || (
    echo [ERROR] cmake not found. Install it, e.g.: python -m pip install cmake
    exit /b 1
)

REM ---- 1. Locate OpenVINO install root (pip site-packages/openvino) ----
for /f "delims=" %%i in ('python -c "import openvino,os;print(os.path.dirname(openvino.__file__))" 2^>nul') do set "OPENVINO_ROOT=%%i"
if not defined OPENVINO_ROOT (
    echo [ERROR] OpenVINO not found. Run: python -m pip install openvino
    exit /b 1
)
echo [INFO] OPENVINO_ROOT=%OPENVINO_ROOT%

REM ---- 2. CMake configure + build (MSVC Release) ----
cmake -S . -B build-py -G "Visual Studio 17 2022" ^
      -DOPENVINO_BACKEND=ON ^
      -DPYTHON_BACKEND=ON ^
      -DOPENVINO_ROOT="%OPENVINO_ROOT%" ^
      -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build-py --config Release || exit /b 1

REM ---- 3. Collect artifacts into fastyolo/libs/ ----
python python\scripts\collect_artifacts.py build-py --backend openvino || exit /b 1

echo.
echo [OK] Build finished. Next steps:
echo     cd python
echo     python -m pytest tests
echo     python examples\demo.py --model model.onnx --image image.jpg
echo     python scripts\build_wheel.py      (optional) make a distributable wheel
endlocal
