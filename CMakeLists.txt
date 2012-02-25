CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(luasrc init.lua 
  display.lua
  dist.lua
  imageio.lua
  sift.lua
  utils.lua
  lena.png
  )

ADD_TORCH_PACKAGE(fex "${src}"  "${luasrc}" "Extra Tools for Feature Extraction")

