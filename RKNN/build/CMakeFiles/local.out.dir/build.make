# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_SOURCE_DIR = /userdata/cai/RSVP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /userdata/cai/RSVP/build

# Include any dependencies generated for this target.
include CMakeFiles/local.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/local.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/local.out.dir/flags.make

CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o: CMakeFiles/local.out.dir/flags.make
CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o: ../src/rknn_local_deploy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/userdata/cai/RSVP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o -c /userdata/cai/RSVP/src/rknn_local_deploy.cpp

CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /userdata/cai/RSVP/src/rknn_local_deploy.cpp > CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.i

CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /userdata/cai/RSVP/src/rknn_local_deploy.cpp -o CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.s

# Object files for target local.out
local_out_OBJECTS = \
"CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o"

# External object files for target local.out
local_out_EXTERNAL_OBJECTS =

local.out: CMakeFiles/local.out.dir/src/rknn_local_deploy.cpp.o
local.out: CMakeFiles/local.out.dir/build.make
local.out: CMakeFiles/local.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/userdata/cai/RSVP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable local.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/local.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/local.out.dir/build: local.out

.PHONY : CMakeFiles/local.out.dir/build

CMakeFiles/local.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/local.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/local.out.dir/clean

CMakeFiles/local.out.dir/depend:
	cd /userdata/cai/RSVP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /userdata/cai/RSVP /userdata/cai/RSVP /userdata/cai/RSVP/build /userdata/cai/RSVP/build /userdata/cai/RSVP/build/CMakeFiles/local.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/local.out.dir/depend
