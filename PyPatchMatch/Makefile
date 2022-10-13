#
# Makefile
# Jiayuan Mao, 2019-01-09 13:59
#

SRC_DIR = csrc
INC_DIR = csrc
OBJ_DIR = build/obj
TARGET = libpatchmatch.so

LIB_TARGET = $(TARGET)
INCLUDE_DIR = -I $(SRC_DIR) -I $(INC_DIR)

CXX = $(ENVIRONMENT_OPTIONS) g++
CXXFLAGS = -std=c++14
CXXFLAGS += -Ofast -ffast-math -w
# CXXFLAGS += -g
CXXFLAGS += $(shell pkg-config --cflags opencv) -fPIC
CXXFLAGS += $(INCLUDE_DIR)
LDFLAGS = $(shell pkg-config --cflags --libs opencv) -shared -fPIC


CXXSOURCES = $(shell find $(SRC_DIR)/ -name "*.cpp")
OBJS = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES:.cpp=.o))
DEPFILES = $(OBJS:.o=.d)

.PHONY: all clean rebuild test

all: $(LIB_TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@echo "[CC] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJ_DIR)/%.d: %.cpp
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cpp=.o) $(OBJ_DIR)/$(<:.cpp=.d)" "$<" > "$@"

sinclude $(DEPFILES)

$(LIB_TARGET): $(OBJS)
	@echo "[link] $(LIB_TARGET) ..."
	@$(CXX) $(OBJS) -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(LIB_TARGET)

rebuild:
	+@make clean
	+@make

# vim:ft=make
#
