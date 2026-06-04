@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation (Windows)
REM Usage:
REM   make.bat html       -> build HTML docs
REM   make.bat clean      -> remove build directory
REM   make.bat latexpdf   -> build PDF (requires LaTeX)

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.ERROR: 'sphinx-build' command was not found.
	echo.Make sure Sphinx is installed and on your PATH:
	echo.
	echo.   pip install sphinx furo
	echo.
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "latexpdf" goto latexpdf

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
echo Build directory removed.
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
echo.
echo.Build finished. Open build\html\index.html in your browser.
goto end

:latexpdf
%SPHINXBUILD% -b latex %SOURCEDIR% %BUILDDIR%/latex %SPHINXOPTS% %O%
cd %BUILDDIR%/latex
make all-pdf
cd %~dp0
echo.PDF built at build\latex\OOPAO.pdf
goto end

:end
popd
