# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/103/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/103/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/artysh/lab_opp/lab_opp2/withoutPar

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/withoutPar.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/withoutPar.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/withoutPar.dir/flags.make

CMakeFiles/withoutPar.dir/main.c.o: CMakeFiles/withoutPar.dir/flags.make
CMakeFiles/withoutPar.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/withoutPar.dir/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/withoutPar.dir/main.c.o   -c /home/artysh/lab_opp/lab_opp2/withoutPar/main.c

CMakeFiles/withoutPar.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/withoutPar.dir/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/artysh/lab_opp/lab_opp2/withoutPar/main.c > CMakeFiles/withoutPar.dir/main.c.i

CMakeFiles/withoutPar.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/withoutPar.dir/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/artysh/lab_opp/lab_opp2/withoutPar/main.c -o CMakeFiles/withoutPar.dir/main.c.s

# Object files for target withoutPar
withoutPar_OBJECTS = \
"CMakeFiles/withoutPar.dir/main.c.o"

# External object files for target withoutPar
withoutPar_EXTERNAL_OBJECTS =

withoutPar: CMakeFiles/withoutPar.dir/main.c.o
withoutPar: CMakeFiles/withoutPar.dir/build.make
withoutPar: CMakeFiles/withoutPar.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable withoutPar"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/withoutPar.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/withoutPar.dir/build: withoutPar

.PHONY : CMakeFiles/withoutPar.dir/build

CMakeFiles/withoutPar.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/withoutPar.dir/cmake_clean.cmake
.PHONY : CMakeFiles/withoutPar.dir/clean

CMakeFiles/withoutPar.dir/depend:
	cd /home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/artysh/lab_opp/lab_opp2/withoutPar /home/artysh/lab_opp/lab_opp2/withoutPar /home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug /home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug /home/artysh/lab_opp/lab_opp2/withoutPar/cmake-build-debug/CMakeFiles/withoutPar.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/withoutPar.dir/depend

