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
include vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/progress.make

# Include the compile flags for this target's objects.
include vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/flags.make

vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/flags.make
vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/includes_CXX.rsp
vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/types/bad_any_cast.cc
vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o -MF CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o.d -o CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/types/bad_any_cast.cc

vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/types/bad_any_cast.cc > CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.i

vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/types/bad_any_cast.cc -o CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.s

# Object files for target absl_bad_any_cast_impl
absl_bad_any_cast_impl_OBJECTS = \
"CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o"

# External object files for target absl_bad_any_cast_impl
absl_bad_any_cast_impl_EXTERNAL_OBJECTS =

vendor/dawn/third_party/abseil-cpp/absl/types/libabsl_bad_any_cast_impl.a: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/bad_any_cast.cc.o
vendor/dawn/third_party/abseil-cpp/absl/types/libabsl_bad_any_cast_impl.a: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/build.make
vendor/dawn/third_party/abseil-cpp/absl/types/libabsl_bad_any_cast_impl.a: vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_bad_any_cast_impl.a"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && $(CMAKE_COMMAND) -P CMakeFiles/absl_bad_any_cast_impl.dir/cmake_clean_target.cmake
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_bad_any_cast_impl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/build: vendor/dawn/third_party/abseil-cpp/absl/types/libabsl_bad_any_cast_impl.a
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/build

vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/clean:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types && $(CMAKE_COMMAND) -P CMakeFiles/absl_bad_any_cast_impl.dir/cmake_clean.cmake
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/clean

vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/depend:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phantom/Developer/Rain2/Rain /home/phantom/Developer/Rain2/Rain/vendor/dawn/third_party/abseil-cpp/absl/types /home/phantom/Developer/Rain2/Rain/build/emscripten-release /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : vendor/dawn/third_party/abseil-cpp/absl/types/CMakeFiles/absl_bad_any_cast_impl.dir/depend

