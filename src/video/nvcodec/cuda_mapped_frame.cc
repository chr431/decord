/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_mapped_frame.cc
 * \brief NVCUVID mapped frame
 */

#include "nv_gpu_dyn.h"
#include "cuda_mapped_frame.h"
#include "../../runtime/cuda/cuda_common.h"
#include <dmlc/logging.h>

namespace decord {
namespace cuda {
using namespace runtime;

CUMappedFrame::CUMappedFrame()
    : disp_info{nullptr}, valid_{false} {
}

CUMappedFrame::CUMappedFrame(CUVIDPARSERDISPINFO* disp_info,
                                    CUvideodecoder decoder,
                                    CUstream stream)
    : disp_info{disp_info}, valid_{false}, decoder_(decoder), params_{0} {

    // Interlaced content is deinterlaced by cuvidMapVideoFrame according to
    // the decoder's DeinterlaceMode (cudaVideoDeinterlaceMode_Adaptive, see
    // cuda_decoder_impl.cc): passing progressive_frame=0 triggers adaptive
    // deinterlacing, while progressive frames pass through untouched.
    params_.progressive_frame = disp_info->progressive_frame;
    params_.top_field_first = disp_info->top_field_first;
    params_.second_field = 0;
    params_.output_stream = stream;

    if (!CHECK_CUDA_CALL(nv::cuvidMapVideoFrame64(decoder_, disp_info->picture_index,
                                   reinterpret_cast<unsigned long long*>(&ptr_), &pitch_, &params_))) {
        LOG(FATAL) << "Unable to map video frame";
    }
    valid_ = true;
}

CUMappedFrame::CUMappedFrame(CUMappedFrame&& other)
    : disp_info{other.disp_info}, valid_{other.valid_}, decoder_{other.decoder_},
      ptr_{other.ptr_}, pitch_{other.pitch_}, params_{other.params_} {
    other.disp_info = nullptr;
    other.valid_ = false;
}

CUMappedFrame::~CUMappedFrame() {
    if (valid_) {
        if (!CHECK_CUDA_CALL(nv::cuvidUnmapVideoFrame64(
                decoder_, static_cast<unsigned long long>(ptr_)))) {
            LOG(FATAL) << "Error unmapping video frame";
        }
    }
}

uint8_t* CUMappedFrame::get_ptr() const {
    return reinterpret_cast<uint8_t*>(ptr_);
}

unsigned int CUMappedFrame::get_pitch() const {
    return pitch_;
}

}  // namespace cuda
}  // namespace decord
