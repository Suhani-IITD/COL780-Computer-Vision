# Makefile for AR Tag Detector (C++)

# Compiler
CXX = g++

# OpenCV paths (Adjust these to your specific installation if different)
# Based on your CMake output, OpenCV is found in C:/msys64/mingw64
OPENCV_DIR = C:/msys64/mingw64
OPENCV_INCLUDE_DIR = $(OPENCV_DIR)/include/opencv4
OPENCV_LIB_DIR = $(OPENCV_DIR)/lib

# Include directories
# Add your custom header directory as well (where image_processing.h and test_functions.h are)
INCLUDE_DIRS = -I$(OPENCV_INCLUDE_DIR) -I.

# Libraries
# Assuming you have a combined OpenCV world library, e.g., opencv_world4100
# Adjust the version number (4100) if your OpenCV version is different
LIBS = -L$(OPENCV_LIB_DIR) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lws2_32 -lgdi32 -luser32 -lkernel32 -lcomdlg32

# Compiler flags
CXXFLAGS = -Wall -std=c++11

# Executable name
TARGET = ar_tag_detector.exe

# Source files
SRCS = main.cpp task1.cpp test_functions.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	del $(OBJS) $(TARGET)

run: clean all
	.\$(TARGET) Tag0.mp4
