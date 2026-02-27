@echo off
type NUL > CON
xcopy /d /y c:\ws\qlib\*.bat
xcopy /d /y c:\ws\qlib\*.py
xcopy /d /y c:\ws\qlib\*.ipynb
xcopy /d /s /y c:\ws\qlib\baselines\* baselines\
xcopy /d /s /y c:\ws\qlib\stability\*.py stability\
xcopy /d /s /y c:\ws\qlib\pipeline\*.py pipeline\