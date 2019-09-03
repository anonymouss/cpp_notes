% run this script to build project on windows with VS2019 %

@echo off
mkdir win_build
cd win_build
cmake -G "Visual Studio 16 2019" ..
cmake --build .
cmake --build . --config Release