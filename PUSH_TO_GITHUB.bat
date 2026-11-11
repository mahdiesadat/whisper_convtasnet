@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ==========================================================
echo   Push this repo to GitHub  (token stays on this PC)
echo ==========================================================
echo.

set /p GHUSER=1) GitHub username: 
set /p REPONAME=2) Repo name (must already exist on github.com): 
set /p TOKEN=3) Paste token (ghp_...), right-click to paste: 

echo.
echo --- checking git ---
git --version
if errorlevel 1 (echo [ERROR] git not found in PATH. & goto :err)

echo --- setting a local identity if missing ---
git config user.email >nul 2>&1
if errorlevel 1 git config user.email "student@example.com"
git config user.name >nul 2>&1
if errorlevel 1 git config user.name "%GHUSER%"

echo --- init / add / commit ---
if not exist ".git" git init
git add -A
if errorlevel 1 (echo [ERROR] git add failed & goto :err)
git commit -m "Sidecar reproduction + P3 experiments: code, results, report"
if errorlevel 1 echo [warn] nothing new to commit (continuing)

git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://%GHUSER%:%TOKEN%@github.com/%GHUSER%/%REPONAME%.git

echo --- pushing ---
git push -u origin main
set RC=%errorlevel%
git remote set-url origin https://github.com/%GHUSER%/%REPONAME%.git
if not "%RC%"=="0" (echo. & echo [FAILED] push returned %RC% & goto :err)

echo.
echo ==========================================================
echo  DONE -> https://github.com/%GHUSER%/%REPONAME%
echo ==========================================================
pause
exit /b 0

:err
echo.
echo ==========================================================
echo  STOPPED. Copy the last 5 lines above and send them.
echo  Common causes:
echo    Authentication failed  -^> wrong/expired token, or token lacks "repo" scope
echo    repository not found   -^> repo name typo, or the repo was not created
echo ==========================================================
pause
exit /b 1
