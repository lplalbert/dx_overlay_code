@echo off
chcp 65001 >nul
echo [DEPRECATED] build_run.bat now delegates to the canonical release build.
echo              Use build_release.bat directly for all supported builds.
call "%~dp0build_release.bat" %*
exit /b %errorlevel%
