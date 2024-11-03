# Install script for directory: /home/phantom/Developer/Rain2/Rain/vendor/assimp/code

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/phantom/Developer/emsdk/upstream/emscripten/cache/sysroot")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libassimp5.4.0-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/assimp/lib/libassimp.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp" TYPE FILE FILES
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/anim.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/aabb.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ai_assert.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/camera.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/color4.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/color4.inl"
    "/home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/assimp/code/../include/assimp/config.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ColladaMetaData.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/commonMetaData.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/defs.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/cfileio.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/light.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/material.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/material.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/matrix3x3.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/matrix3x3.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/matrix4x4.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/matrix4x4.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/mesh.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ObjMaterial.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/pbrmaterial.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/GltfMaterial.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/postprocess.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/quaternion.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/quaternion.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/scene.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/metadata.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/texture.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/types.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/vector2.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/vector2.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/vector3.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/vector3.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/version.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/cimport.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/AssertHandler.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/importerdesc.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Importer.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/DefaultLogger.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ProgressHandler.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/IOStream.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/IOSystem.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Logger.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/LogStream.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/NullLogger.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/cexport.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Exporter.hpp"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/DefaultIOStream.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/DefaultIOSystem.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ZipArchiveIOSystem.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SceneCombiner.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/fast_atof.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/qnan.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/BaseImporter.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Hash.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/MemoryIOWrapper.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ParsingUtils.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/StreamReader.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/StreamWriter.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/StringComparison.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/StringUtils.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SGSpatialSort.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/GenericProperty.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SpatialSort.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SkeletonMeshBuilder.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SmallVector.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SmoothingGroups.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/SmoothingGroups.inl"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/StandardShapes.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/RemoveComments.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Subdivision.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Vertex.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/LineSplitter.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/TinyFormatter.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Profiler.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/LogAux.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Bitmap.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/XMLTools.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/IOStreamBuffer.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/CreateAnimMesh.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/XmlParser.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/BlobIOSystem.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/MathFunctions.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Exceptional.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/ByteSwapper.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Base64.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp/Compiler" TYPE FILE FILES
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Compiler/pushpack1.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Compiler/poppack1.h"
    "/home/phantom/Developer/Rain2/Rain/vendor/assimp/code/../include/assimp/Compiler/pstdint.h"
    )
endif()

