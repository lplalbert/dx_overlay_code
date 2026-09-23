@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "NO_PAUSE=0"
if /i "%~1"=="--no-pause" set "NO_PAUSE=1"

echo ========================================
echo   DX Overlay - Release Build
echo   (Self-contained Overlay + Generator)
echo ========================================
echo.

set "VERSION=v4.2"
set "PROJECT_DIR=%~dp0"
set "RELEASE_DIR=%PROJECT_DIR%release_dist"
set "EXE_NAME=dx_overlay.exe"
set "BUILD_EXE_NAME=dx_overlay_build_tmp.exe"
set "GENERATOR_EXE_NAME=wm_generator.exe"
set "PYI_DIST_DIR=%PROJECT_DIR%build_release_dist"
set "PYI_WORK_DIR=%PROJECT_DIR%build_release_py"
set "ZIP_NAME=dx_overlay_%VERSION%_release.zip"

pushd "%PROJECT_DIR%" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Cannot enter project directory: %PROJECT_DIR%
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo [Build Info]
echo   Version: %VERSION%
echo   Output : %RELEASE_DIR%
echo   Flow   : config.txt -^> wm_generator.exe -^> wm_imgs -^> dx_overlay.exe
echo.

where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] g++ not found in PATH.
    echo         Please install MinGW-w64 and add it to PATH.
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python not found in PATH.
    echo         Python is required to package wm_generator.exe for release.
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

python -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller is not available.
    echo         Run: pip install -r requirements.txt
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

echo [1/6] Preparing release directory...
for %%F in ("%RELEASE_DIR%\%EXE_NAME%" "%RELEASE_DIR%\dx_overlay.log") do (
    if exist "%%~fF" (
        powershell -NoProfile -Command "try { $s = [System.IO.File]::Open('%%~fF', 'Open', 'ReadWrite', 'None'); $s.Dispose(); exit 0 } catch { exit 1 }"
        if errorlevel 1 goto :release_directory_locked
    )
)
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
if exist "%RELEASE_DIR%" goto :release_directory_locked
mkdir "%RELEASE_DIR%" || goto :release_prepare_failed
mkdir "%RELEASE_DIR%\wm_imgs" || goto :release_prepare_failed
mkdir "%RELEASE_DIR%\tools" || goto :release_prepare_failed

echo [2/6] Compiling C++ overlay...
if exist "%PROJECT_DIR%%BUILD_EXE_NAME%" del /q "%PROJECT_DIR%%BUILD_EXE_NAME%"
g++ -std=c++17 -O2 -DUNICODE -D_UNICODE -mwindows ^
    -static-libgcc -static-libstdc++ ^
    -o %BUILD_EXE_NAME% ^
    main.cpp overlay.cpp overlay_window.cpp overlay_device.cpp ^
    overlay_texture.cpp overlay_render.cpp overlay_recovery.cpp ^
    -ld3d11 -ldxgi -ld3dcompiler -ldcomp -lole32 -lwindowscodecs -lshcore -lshell32

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

echo [OK] %BUILD_EXE_NAME% compiled.
echo.

echo [3/6] Compiling diagnostic tools...
g++ -std=c++17 -O2 -DUNICODE -D_UNICODE ^
    -static-libgcc -static-libstdc++ ^
    -o "%RELEASE_DIR%\tools\test_alpha.exe" test_alpha.cpp
if %errorlevel% neq 0 goto :tool_build_failed

g++ -std=c++17 -O2 -DUNICODE -D_UNICODE ^
    -static-libgcc -static-libstdc++ ^
    -o "%RELEASE_DIR%\tools\test_frame3.exe" test_frame3.cpp
if %errorlevel% neq 0 goto :tool_build_failed

g++ -std=c++17 -O2 -DUNICODE -D_UNICODE ^
    -static-libgcc -static-libstdc++ ^
    -o "%RELEASE_DIR%\tools\test_stability.exe" test_stability.cpp -lpsapi
if %errorlevel% neq 0 goto :tool_build_failed
echo [OK] Diagnostic tools compiled.
echo.

echo [4/6] Packaging watermark generator...
if exist "%PYI_DIST_DIR%" rmdir /s /q "%PYI_DIST_DIR%"
if exist "%PYI_WORK_DIR%" rmdir /s /q "%PYI_WORK_DIR%"

python -m PyInstaller ^
    --onefile ^
    --console ^
    --name %GENERATOR_EXE_NAME:.exe=% ^
    --distpath "%PYI_DIST_DIR%" ^
    --workpath "%PYI_WORK_DIR%" ^
    --specpath "%PYI_WORK_DIR%" ^
    --clean ^
    --noconfirm ^
    --hidden-import reedsolo ^
    --collect-all cv2 ^
    rs_gen_Syn_template_nums_dual.py

if %errorlevel% neq 0 (
    echo [ERROR] Failed to package %GENERATOR_EXE_NAME%.
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

if not exist "%PYI_DIST_DIR%\%GENERATOR_EXE_NAME%" (
    echo [ERROR] %GENERATOR_EXE_NAME% was not created.
    if "%NO_PAUSE%"=="0" pause
    popd
    exit /b 1
)

echo [OK] %GENERATOR_EXE_NAME% packaged.
echo.

echo [5/6] Copying runtime files...
copy /y "%PROJECT_DIR%%BUILD_EXE_NAME%" "%RELEASE_DIR%\%EXE_NAME%" >nul || goto :runtime_copy_failed
copy /y "%PYI_DIST_DIR%\%GENERATOR_EXE_NAME%" "%RELEASE_DIR%\%GENERATOR_EXE_NAME%" >nul || goto :runtime_copy_failed
copy /y "%PROJECT_DIR%config.txt" "%RELEASE_DIR%\" >nul || goto :runtime_copy_failed
copy /y "%PROJECT_DIR%requirements.txt" "%RELEASE_DIR%\" >nul || goto :runtime_copy_failed
copy /y "%PROJECT_DIR%rs_gen_Syn_template_nums_dual.py" "%RELEASE_DIR%\" >nul || goto :runtime_copy_failed

echo Creating README.txt...
(
echo ==========================================
echo   DX Overlay - Screen Watermark Tool %VERSION%
echo ==========================================
echo.
echo [Workflow]
echo   1. Edit config.txt
echo   2. Double-click dx_overlay.exe
echo   3. The app will use the bundled wm_generator.exe
echo      to generate watermark templates into wm_imgs\
echo      and then open the transparent overlay window.
echo.
echo [Required Files]
echo   dx_overlay.exe
echo   wm_generator.exe
echo   config.txt
echo   requirements.txt
echo   rs_gen_Syn_template_nums_dual.py
echo   wm_imgs\
echo.
echo [Important Config]
echo   static_alpha     - Static Cb runtime strength ^(0.0-1.0^)
echo   dynamic_alpha    - Dynamic Cr runtime strength ^(0.0-1.0^)
echo   alpha            - Legacy fallback that sets both strengths
echo   monitor_index    - Target monitor ^(0 primary, 1 secondary, -1 virtual desktop^)
echo   switch_hold_frames - Target display refreshes to hold each texture
echo   transition_frames  - Display-refresh intervals per switch ^(1 hard switch^)
echo   watermark_id     - Numeric watermark ID
echo   block_rows       - Watermark block rows
echo   block_cols       - Watermark block columns
echo   dynamic_ratio_cr - Dynamic Cr generator amplitude ^(legacy ratio_u^)
echo   static_ratio_cb  - Static Cb generator amplitude ^(legacy ratio_v^)
echo   type_val         - Pattern type
echo   Watermark spreading grid is fixed at 8x8
echo   pattern          - Static Cb shape: gaussian / gaussian_v2 / soft_rect / rect
echo   train_codeword   - Optional training mode ^(-1 disables^)
echo   generator_mode   - auto / rs_script / packed_exe / slim_script
echo   python_path      - Python interpreter path
echo.
echo [Current Release Behavior]
echo   - generator_mode=auto will prefer wm_generator.exe when available
echo   - runtime auto-aligns watermark size to the selected monitor resolution
echo   - generated files use the names:
echo       wm_template_XXX.png
echo       wm_template_XXX_inverse.png
echo   - templates encode R=Y, G=dynamic Cr, B=static Cb, A=254 format marker
echo   - tools\test_alpha.exe reads both shader channel strengths
echo   - tools\test_frame3.exe compares Present submissions with DXGI displayed-frame statistics
echo   - tools\test_stability.exe monitors resource usage
echo.
echo [Install Dependencies ^(Development / fallback only^)]
echo   pip install -r requirements.txt
echo.
echo [Troubleshooting]
echo   1. Run dx_overlay.exe log to create dx_overlay.log
echo   2. Check that wm_generator.exe exists next to dx_overlay.exe
echo   3. If you intentionally switch to script mode, install requirements.txt
echo   4. Ensure the target monitor is connected before launch
echo.
echo ==========================================
) > "%RELEASE_DIR%\README.txt"
if errorlevel 1 goto :runtime_copy_failed

echo [6/6] Creating zip package...
if exist "%PROJECT_DIR%%ZIP_NAME%" del /q "%PROJECT_DIR%%ZIP_NAME%"
if exist "%PROJECT_DIR%%ZIP_NAME%" goto :zip_failed
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath '%PROJECT_DIR%%ZIP_NAME%' -Force"

if errorlevel 1 goto :zip_failed
if not exist "%PROJECT_DIR%%ZIP_NAME%" goto :zip_failed
echo [OK] Package created: %ZIP_NAME%

echo.
echo ========================================
echo   Release Build Complete
echo ========================================
echo.
echo Release directory: %RELEASE_DIR%
echo Package: %PROJECT_DIR%%ZIP_NAME%
echo.
echo Included:
echo   - dx_overlay.exe
echo   - wm_generator.exe
echo   - config.txt
echo   - requirements.txt
echo   - rs_gen_Syn_template_nums_dual.py
echo   - wm_imgs\
echo   - tools\test_alpha.exe
echo   - tools\test_frame3.exe
echo   - tools\test_stability.exe
echo.
echo Next step for users:
echo   double-click dx_overlay.exe
echo.

if exist "%PROJECT_DIR%%BUILD_EXE_NAME%" del /q "%PROJECT_DIR%%BUILD_EXE_NAME%" >nul 2>&1
if exist "%PYI_DIST_DIR%" rmdir /s /q "%PYI_DIST_DIR%" >nul 2>&1
if exist "%PYI_WORK_DIR%" rmdir /s /q "%PYI_WORK_DIR%" >nul 2>&1
if "%NO_PAUSE%"=="0" pause
popd
endlocal
exit /b 0

:tool_build_failed
echo [ERROR] Failed to compile diagnostic tools.
goto :build_failed

:release_directory_locked
echo [ERROR] Cannot replace %RELEASE_DIR%.
echo         Close dx_overlay.exe or any program using files in that directory, then retry.
goto :build_failed

:release_prepare_failed
echo [ERROR] Failed to prepare release directory: %RELEASE_DIR%
goto :build_failed

:runtime_copy_failed
echo [ERROR] Failed to copy or create one or more release runtime files.
goto :build_failed

:zip_failed
echo [ERROR] Failed to create a complete zip package: %PROJECT_DIR%%ZIP_NAME%
goto :build_failed

:build_failed
if "%NO_PAUSE%"=="0" pause
if exist "%PROJECT_DIR%%BUILD_EXE_NAME%" del /q "%PROJECT_DIR%%BUILD_EXE_NAME%" >nul 2>&1
if exist "%PYI_DIST_DIR%" rmdir /s /q "%PYI_DIST_DIR%" >nul 2>&1
if exist "%PYI_WORK_DIR%" rmdir /s /q "%PYI_WORK_DIR%" >nul 2>&1
popd
endlocal
exit /b 1
