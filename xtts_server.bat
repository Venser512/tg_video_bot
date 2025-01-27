call xtts\\Scripts\\activate 
python -m xtts_api_server --bat-dir %~dp0 -d=cuda --deepspeed
call xtts\\Scripts\\activate
pause