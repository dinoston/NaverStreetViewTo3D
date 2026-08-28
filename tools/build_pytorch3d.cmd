@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "CONDA_ENV=%PROJECT_ROOT%\tools\gs-env"
set "CUDA_HOME=%CONDA_ENV%"
set "PATH=%CONDA_ENV%;%CONDA_ENV%\bin;%CONDA_ENV%\Library;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%PATH%"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "FORCE_CUDA=1"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=8"

call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%

set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
"%CONDA_ENV%\python.exe" -m pip install --no-build-isolation "%PROJECT_ROOT%\external\pytorch3d"
exit /b %errorlevel%
