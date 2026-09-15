# Use paths relative to Linux home directory
TFLM_PATH := $(HOME)/tflite-micro
TFLM_DL := $(TFLM_PATH)/tensorflow/lite/micro/tools/make/downloads

CXX := g++
CXXFLAGS := -O3 -std=c++17 -DTF_LITE_STATIC_MEMORY
INCLUDES := -I. -I$(TFLM_PATH) -I$(TFLM_DL)/flatbuffers/include -I$(TFLM_DL)/gemmlowp -I$(TFLM_DL)/ruy

TARGET := micro_inference_app
SRCS := part3_micro_deployment.cc model_data.cc
OBJS := $(SRCS:.cc=.o)

LIBS := -L$(TFLM_PATH)/gen/linux_x86_64_default_gcc/lib -ltensorflow-microlite

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)