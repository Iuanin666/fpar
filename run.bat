@echo off
set PATH=E:\anaconda\Library\bin;%PATH%

:menu
cls
echo =========================================================
echo FPAR Project Runner - V7 Cross-Scale Version
echo =========================================================
echo [Data Prep]
echo 0. Run MODIS Alignment (1000m -^> 500m)
echo.
echo [Baseline Models]
echo 1. Run U-Net Training
echo 2. Run Transformer Training
echo 3. Run U-Net Evaluation
echo 4. Run Transformer Evaluation
echo.
echo [V7 Cross-Scale Model]
echo 5. Run CrossScale Training (Standard)
echo 6. Run CrossScale Training (Resume)
echo 7. Run CrossScale Training (3-epoch Test)
echo 8. Run CrossScale Evaluation
echo.
echo [TensorBoard]
echo 9. Start TensorBoard (All Logs)
echo.
echo 10. Exit
echo =========================================================
echo.

choice /c 0123456789X /n /m "Select an option (0-9, X for exit): "

if errorlevel 11 goto end
if errorlevel 10 goto tensorboard_all
if errorlevel 9 goto evaluate_cs
if errorlevel 8 goto train_cs_test
if errorlevel 7 goto train_cs_resume
if errorlevel 6 goto train_cs
if errorlevel 5 goto evaluate_trans
if errorlevel 4 goto evaluate_unet
if errorlevel 3 goto train_trans
if errorlevel 2 goto train_unet
if errorlevel 1 goto align_modis

:align_modis
echo Starting MODIS Alignment...
E:\anaconda\envs\fpar_project\python.exe src\7_align_modis.py
pause
goto menu

:train_unet
echo Starting U-Net training...
E:\anaconda\envs\fpar_project\python.exe src\4_train.py
pause
goto menu

:train_trans
echo Starting Transformer training...
E:\anaconda\envs\fpar_project\python.exe src\6_train_transformer.py
pause
goto menu

:evaluate_unet
echo Starting U-Net evaluation...
E:\anaconda\envs\fpar_project\python.exe src\evaluate.py --model unet
pause
goto menu

:evaluate_trans
echo Starting Transformer evaluation...
E:\anaconda\envs\fpar_project\python.exe src\evaluate.py --model transformer
pause
goto menu

:train_cs
echo Starting CrossScale standard training...
E:\anaconda\envs\fpar_project\python.exe src\9_train_crossscale.py
pause
goto menu

:train_cs_resume
echo Resuming CrossScale training from checkpoint...
E:\anaconda\envs\fpar_project\python.exe src\9_train_crossscale.py --resume
pause
goto menu

:train_cs_test
echo Starting CrossScale mini-test (3 epochs)...
E:\anaconda\envs\fpar_project\python.exe src\9_train_crossscale.py --test_epochs 3
pause
goto menu

:evaluate_cs
echo Starting CrossScale evaluation...
E:\anaconda\envs\fpar_project\python.exe src\evaluate.py --model crossscale
pause
goto menu

:tensorboard_all
echo Starting TensorBoard on http://localhost:6006...
start "" "http://localhost:6006"
E:\anaconda\envs\fpar_project\Scripts\tensorboard.exe --logdir=E:\FPAR_project
pause
goto menu

:end
