#!/bin/bash

cd $HOME
git clone --recursive https://github.com/boostorg/boost.git
cd boost
./bootstrap.sh --prefix=/usr/local
./b2 headers # http://boost.2283326.n4.nabble.com/Fwd-b2-install-not-copying-all-headers-td4692176.html
sudo ./b2 install --libdir=/usr/local/lib/boost --includedir=/usr/local/include/boost toolset=clang
cd $HOME