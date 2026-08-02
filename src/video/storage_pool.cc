/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file storage_pool.cc
 * \brief Simple pool for storage
 */

#include "storage_pool.h"

namespace decord {

NDArrayPool::NDArrayPool() : init_(false) {

}

NDArrayPool::NDArrayPool(std::size_t sz, std::vector<int64_t> shape, DLDataType dtype, DLDevice ctx)
    : size_(sz), shape_(shape), dtype_(dtype), ctx_(ctx), init_(true) {
}

NDArrayPool::~NDArrayPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (queue_.size() > 0) {
        auto arr = queue_.front();
        queue_.pop();
        arr.data_->manager_ctx = nullptr;
    }
}

void NDArrayPool::Reset(std::size_t sz, std::vector<int64_t> shape, DLDataType dtype, DLDevice ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    // release any pooled buffers (they recycle through the deleter)
    while (queue_.size() > 0) {
        auto arr = queue_.front();
        queue_.pop();
        arr.data_->manager_ctx = nullptr;
    }
    size_ = sz;
    shape_ = shape;
    dtype_ = dtype;
    ctx_ = ctx;
    init_ = true;
}

runtime::NDArray NDArrayPool::Acquire() {
    CHECK(init_) << "NDArrayPool not initialized with shape and ctx";
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() > 0) {
            auto arr = queue_.front();
            queue_.pop();
            return arr;
        }
    }
    {
        // Allocate
        auto arr = NDArray::Empty(shape_, dtype_, ctx_);
        arr.data_->manager_ctx = this;
        arr.data_->deleter = &NDArrayPool::Deleter;
        return arr;
    }
}

void NDArrayPool::Deleter(NDArray::Container* ptr) {
    if (!ptr) return;
    if (ptr->manager_ctx != nullptr) {
        auto pool = static_cast<NDArrayPool*>(ptr->manager_ctx);
        {
            std::lock_guard<std::mutex> lock(pool->mutex_);
            if (pool->size_ <= pool->queue_.size()) {
                decord::runtime::DeviceAPI::Get(ptr->dl_tensor.device)->FreeDataSpace(
                    ptr->dl_tensor.device, ptr->dl_tensor.data);
                delete ptr;
                ptr = nullptr;
            } else {
                pool->queue_.push(NDArray(ptr));
            }
        }
    } else if (ptr->dl_tensor.data != nullptr) {
        decord::runtime::DeviceAPI::Get(ptr->dl_tensor.device)->FreeDataSpace(
          ptr->dl_tensor.device, ptr->dl_tensor.data);
        delete ptr;
        ptr = nullptr;
    }
}

}  // namespace decord
