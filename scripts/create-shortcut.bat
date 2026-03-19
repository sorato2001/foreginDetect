@echo off
REM Windows Command Prompt Helper - Creates shortcut for easier startup
REM Double-click this file to create a desktop shortcut for run.bat

setlocal enabledelayedexpansion

REM Get this script's directory
set SCRIPT_DIR=%~dp0

REM Create shortcut to run.bat
set SHORTCUT_PATH=%USERPROFILE%\Desktop\Start-MultiCamera.lnk
set TARGET=%SCRIPT_DIR%run.bat

powershell -Command ^
  "$WshShell = New-Object -ComObject WScript.Shell; " ^
  "$Shortcut = $WshShell.CreateShortCut('%SHORTCUT_PATH%'); " ^
  "$Shortcut.TargetPath = '%TARGET%'; " ^
  "$Shortcut.WorkingDirectory = '%SCRIPT_DIR%..'; " ^
  "$Shortcut.Description = 'Multi-Camera Event Recording System'; " ^
  "$Shortcut.Save()"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Desktop shortcut created:
    echo   %SHORTCUT_PATH%
    echo.
    echo You can now double-click the shortcut to start the system.
    echo.
) else (
    echo ERROR: Failed to create shortcut
)

pause
