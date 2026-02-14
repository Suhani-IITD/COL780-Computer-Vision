# Makefile for AR Tag Detector (C++)

CXX ?= g++
CXXFLAGS ?= -Wall -std=c++11

TARGET = ar_tag_detector
SRCS = main.cpp task1.cpp
OBJS = $(SRCS:.cpp=.o)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

ifeq ($(strip $(OPENCV_LIBS)),)
$(error OpenCV4 not found via pkg-config. Install OpenCV dev package or edit Makefile with local include/library paths.)
endif

CPPFLAGS += $(OPENCV_CFLAGS)
LDFLAGS += $(OPENCV_LIBS)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: all
	./$(TARGET) Tag0.mp4
