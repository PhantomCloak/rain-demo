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
include vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/compiler_depend.make

# Include the progress variables for this target.
include vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/progress.make

# Include the compile flags for this target's objects.
include vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/flags.make

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/flags.make
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/includes_CXX.rsp
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/clone_context.cc
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o -MF CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o.d -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/clone_context.cc

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/clone_context.cc > CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.i

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/clone_context.cc -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.s

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/flags.make
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/includes_CXX.rsp
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program.cc
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o -MF CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o.d -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program.cc

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program.cc > CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.i

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program.cc -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.s

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/flags.make
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/includes_CXX.rsp
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o: /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program_builder.cc
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o -MF CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o.d -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o -c /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program_builder.cc

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.i"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program_builder.cc > CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.i

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.s"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && /home/phantom/Developer/emsdk/upstream/emscripten/em++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint/lang/wgsl/program/program_builder.cc -o CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.s

# Object files for target tint_lang_wgsl_program
tint_lang_wgsl_program_OBJECTS = \
"CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o" \
"CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o" \
"CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o"

# External object files for target tint_lang_wgsl_program
tint_lang_wgsl_program_EXTERNAL_OBJECTS =

vendor/dawn/src/tint/libtint_lang_wgsl_program.a: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/clone_context.cc.o
vendor/dawn/src/tint/libtint_lang_wgsl_program.a: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program.cc.o
vendor/dawn/src/tint/libtint_lang_wgsl_program.a: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/lang/wgsl/program/program_builder.cc.o
vendor/dawn/src/tint/libtint_lang_wgsl_program.a: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/build.make
vendor/dawn/src/tint/libtint_lang_wgsl_program.a: vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/phantom/Developer/Rain2/Rain/build/emscripten-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libtint_lang_wgsl_program.a"
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -P CMakeFiles/tint_lang_wgsl_program.dir/cmake_clean_target.cmake
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tint_lang_wgsl_program.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/build: vendor/dawn/src/tint/libtint_lang_wgsl_program.a
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/build

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/clean:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint && $(CMAKE_COMMAND) -P CMakeFiles/tint_lang_wgsl_program.dir/cmake_clean.cmake
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/clean

vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/depend:
	cd /home/phantom/Developer/Rain2/Rain/build/emscripten-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phantom/Developer/Rain2/Rain /home/phantom/Developer/Rain2/Rain/vendor/dawn/src/tint /home/phantom/Developer/Rain2/Rain/build/emscripten-release /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint /home/phantom/Developer/Rain2/Rain/build/emscripten-release/vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : vendor/dawn/src/tint/CMakeFiles/tint_lang_wgsl_program.dir/depend

