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
include vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/progress.make

# Include the compile flags for this target's objects.
include vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/flags.make

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/flags.make
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/includes_CXX.rsp
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/generate_external_texture_bindings.cc
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o -MF CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o.d -o CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/generate_external_texture_bindings.cc

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/generate_external_texture_bindings.cc > CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.i

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/generate_external_texture_bindings.cc -o CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.s

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/flags.make
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/includes_CXX.rsp
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/helper.cc
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o -MF CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o.d -o CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/helper.cc

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/helper.cc > CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.i

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/cmd/common/helper.cc -o CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.s

# Object files for target tint_cmd_common
tint_cmd_common_OBJECTS = \
"CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o" \
"CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o"

# External object files for target tint_cmd_common
tint_cmd_common_EXTERNAL_OBJECTS =

vendor/dawn/src/tint/libtint_cmd_common.a: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/generate_external_texture_bindings.cc.o
vendor/dawn/src/tint/libtint_cmd_common.a: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/cmd/common/helper.cc.o
vendor/dawn/src/tint/libtint_cmd_common.a: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/build.make
vendor/dawn/src/tint/libtint_cmd_common.a: vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libtint_cmd_common.a"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -P CMakeFiles/tint_cmd_common.dir/cmake_clean_target.cmake
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tint_cmd_common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/build: vendor/dawn/src/tint/libtint_cmd_common.a
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/build

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/clean:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -P CMakeFiles/tint_cmd_common.dir/cmake_clean.cmake
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/clean

vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/depend:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phantom/Developer/Rain2/Rain /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint /home/phantom/Developer/Rain2/Rain/build/emscripten-release /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_cmd_common.dir/depend

