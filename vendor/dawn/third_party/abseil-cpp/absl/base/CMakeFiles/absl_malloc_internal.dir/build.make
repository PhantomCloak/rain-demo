# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/phantom/Developer/Rain2/Rain

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/phantom/Developer/Rain2/Rain/build/emscripten-release

# Include any dependencies generated for this target.
include vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/progress.make

# Include the compile flags for this target's objects.
include vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/flags.make

vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/flags.make
vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/includes_CXX.rsp
vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/base/internal/low_level_alloc.cc
vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o -MF CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o.d -o CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/base/internal/low_level_alloc.cc

vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/base/internal/low_level_alloc.cc > CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.i

vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/base/internal/low_level_alloc.cc -o CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.s

# Object files for target absl_malloc_internal
absl_malloc_internal_OBJECTS = \
"CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o"

# External object files for target absl_malloc_internal
absl_malloc_internal_EXTERNAL_OBJECTS =

vendor/dawn/third_party/abseil-cpp/absl/base/libabsl_malloc_internal.a: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/internal/low_level_alloc.cc.o
vendor/dawn/third_party/abseil-cpp/absl/base/libabsl_malloc_internal.a: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/build.make
vendor/dawn/third_party/abseil-cpp/absl/base/libabsl_malloc_internal.a: vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_malloc_internal.a"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && $(CMAKE_COMMAND) -P CMakeFiles/absl_malloc_internal.dir/cmake_clean_target.cmake
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_malloc_internal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/build: vendor/dawn/third_party/abseil-cpp/absl/base/libabsl_malloc_internal.a
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/build

vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/clean:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base && $(CMAKE_COMMAND) -P CMakeFiles/absl_malloc_internal.dir/cmake_clean.cmake
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/clean

vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/depend:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phantom/Developer/Rain2/Rain /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/base /home/phantom/Developer/Rain2/Rain/build/emscripten-release /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/base/CMakeFiles/absl_malloc_internal.dir/depend

