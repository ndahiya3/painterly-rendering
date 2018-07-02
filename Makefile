CXX        = g++ -g -std=c++11    #For object files
OPENCV_INCLUDE   = -I/usr/include
OPENCV_LIB_PATH = -L/usr/lib64
OPENCV_LIB = ${OPENCV_LIB_PATH} -lopencv_core -lopencv_highgui -lopencv_imgproc

all: painterly
painterly: main.o CurvedStroke.o
	${CXX} -o painterly main.o CurvedStroke.o ${OPENCV_INCLUDE} ${OPENCV_LIB}
clean:
	rm -f main.o CurvedStroke.o painterly
