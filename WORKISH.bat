@echo off
pushd "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File ".\Scripts\Activate.ps1"
popd