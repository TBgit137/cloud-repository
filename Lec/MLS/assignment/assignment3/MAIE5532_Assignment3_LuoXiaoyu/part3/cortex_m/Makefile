TFLM_ROOT ?= /home/mywslubuntu/tflite-micro
TOOLCHAIN_BIN ?= $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin
TFLM_LIB ?= $(TFLM_ROOT)/gen/cortex_m_qemu_cortex-m3_default_gcc/lib/libtensorflow-microlite.a
MODEL ?= ../../part2/part2_results/quantized_models/ptq_int8.tflite
PYTHON ?= python3
QEMU ?= qemu-arm
CXX := $(TOOLCHAIN_BIN)/arm-none-eabi-g++
TARGET ?= qemu_cifar_benchmark

CPPFLAGS := \
	-DTF_LITE_STATIC_MEMORY \
	-DTF_LITE_DISABLE_X86_NEON \
	-DTF_LITE_MCU_DEBUG_LOG \
	-I$(TFLM_ROOT) \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/ruy

CXXFLAGS := \
	-std=c++17 \
	-mcpu=cortex-m3 \
	-mthumb \
	-mfloat-abi=soft \
	-funsigned-char \
	-mlittle-endian \
	-fno-rtti \
	-fno-exceptions \
	-fno-threadsafe-statics \
	-fno-unwind-tables \
	-fno-asynchronous-unwind-tables \
	-ffunction-sections \
	-fdata-sections \
	-O2 \
	--specs=rdimon.specs

LDFLAGS := \
	-Wl,--fatal-warnings \
	-Wl,--gc-sections \
	-lm

.PHONY: all run model analyze clean

all: $(TARGET)

model: generated/model_data.cc

generated/model_data.cc: $(MODEL) ../tools/tflite_to_c.py
	$(PYTHON) ../tools/tflite_to_c.py $(MODEL) $@

$(TARGET): qemu_benchmark.cc generated/model_data.cc
	test -f "$(TFLM_LIB)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		qemu_benchmark.cc generated/model_data.cc \
		$(TFLM_LIB) $(LDFLAGS) -o $@

run: $(TARGET)
	$(QEMU) -cpu cortex-m3 $(TARGET)

analyze:
	cd .. && $(PYTHON) part3_modeling.py

clean:
	rm -f $(TARGET) generated/model_data.cc
