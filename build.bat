
@echo off

REM Record the original directory
set "original_dir=%cd%"


set vcpkg="-DCMAKE_TOOLCHAIN_FILE=C:/Users/axela/Documents/GitHub/vcpkg/scripts/buildsystems/vcpkg.cmake"
set cmake="C:\Program Files\CMake\bin\cmake.exe"
set msvc=-DMSVC=TRUE
set type=-DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF
 
cd build
%cmake% %vcpkg% %msvc% %type% ../  
if NOT %ERRORLEVEL% == 0 goto :endofscript

%cmake% --build . --config RelWithDebInfo
if NOT %ERRORLEVEL% == 0 goto :endofscript

cd ..
@REM "./build/Release/RealTimeCameraProcessing.exe"  
"./build/RelWithDebInfo/RealTimeCameraProcessing.exe"  

:endofscript
@REM if %ERRORLEVEL% == 0 echo "Success!"
if NOT %ERRORLEVEL% == 0 (
    echo "Cmake failed!"
    cd /d "%original_dir%"
    )


