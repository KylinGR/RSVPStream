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
include CMakeFiles/rknn_main.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rknn_main.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rknn_main.out.dir/flags.make

CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o: CMakeFiles/rknn_main.out.dir/flags.make
CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o: ../src/rknn_model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/userdata/cai/RSVP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o -c /userdata/cai/RSVP/src/rknn_model.cpp

CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /userdata/cai/RSVP/src/rknn_model.cpp > CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.i

CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /userdata/cai/RSVP/src/rknn_model.cpp -o CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.s

CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o: CMakeFiles/rknn_main.out.dir/flags.make
CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o: ../src/rknn_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/userdata/cai/RSVP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o -c /userdata/cai/RSVP/src/rknn_main.cpp

CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /userdata/cai/RSVP/src/rknn_main.cpp > CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.i

CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /userdata/cai/RSVP/src/rknn_main.cpp -o CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.s

# Object files for target rknn_main.out
rknn_main_out_OBJECTS = \
"CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o" \
"CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o"

# External object files for target rknn_main.out
rknn_main_out_EXTERNAL_OBJECTS =

rknn_main.out: CMakeFiles/rknn_main.out.dir/src/rknn_model.cpp.o
rknn_main.out: CMakeFiles/rknn_main.out.dir/src/rknn_main.cpp.o
rknn_main.out: CMakeFiles/rknn_main.out.dir/build.make
rknn_main.out: CMakeFiles/rknn_main.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/userdata/cai/RSVP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable rknn_main.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rknn_main.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rknn_main.out.dir/build: rknn_main.out

.PHONY : CMakeFiles/rknn_main.out.dir/build

CMakeFiles/rknn_main.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rknn_main.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rknn_main.out.dir/clean

CMakeFiles/rknn_main.out.dir/depend:
	cd /userdata/cai/RSVP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /userdata/cai/RSVP /userdata/cai/RSVP /userdata/cai/RSVP/build /userdata/cai/RSVP/build /userdata/cai/RSVP/build/CMakeFiles/rknn_main.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rknn_main.out.dir/depend

