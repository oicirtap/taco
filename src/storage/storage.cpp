#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "taco/type.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Storage
struct Storage::Content {
  Format format;
  Index  index;
  Array  values;
  Array  d_values;
};

Storage::Storage() : content(nullptr) {
}

Storage::Storage(const Format& format) : content(new Content) {
  content->format = format;
}

void Storage::setValues(const Array& values) {
  content->values = values;
  void * d_data;
  int numBytes = values.getSize() * values.getType().getNumBytes();
  cudaMalloc(&d_data, numBytes);
  cudaMemcpy(d_data, values.getData(), numBytes, cudaMemcpyHostToDevice);
  content->d_values = Array(values.getType(), d_data, values.getSize(), Array::Free, false);
}

const Format& Storage::getFormat() const {
  return content->format;
}

void Storage::setIndex(const Index& index) {
  content->index = index;
}

const Index& Storage::getIndex() const {
  return content->index;
}

Index Storage::getIndex() {
  return content->index;
}

const Array& Storage::getValues() const {
  return content->values;
}

Array Storage::getValues() {
  return content->values;
}

const Array& Storage::getDValues() const {
  return content->d_values;
}

Array Storage::getDValues() {
  return content->d_values;
}

size_t Storage::getSizeInBytes() {
  size_t indexSizeInBytes = 0;
  const auto& index = getIndex();
  for (size_t i = 0; i < index.numModeIndices(); i++) {
    const auto& modeIndex = index.getModeIndex(i);
    for (size_t j = 0; j < modeIndex.numIndexArrays(); j++) {
      const auto& indexArray = modeIndex.getIndexArray(j);
      indexSizeInBytes += indexArray.getSize() *
                          indexArray.getType().getNumBytes();
    }
  }
  const auto& values = getValues();
  return indexSizeInBytes + values.getSize() * values.getType().getNumBytes();
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  return os << storage.getIndex() << endl << storage.getValues();
}

}}
