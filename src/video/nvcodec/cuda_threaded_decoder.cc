/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.cc
 * \brief NVCUVID based decoder impl
 */

#include "cuda_threaded_decoder.h"
#include "cuda_mapped_frame.h"
#include "cuda_texture.h"
#include "../../improc/improc.h"
#include "nv_gpu_dyn.h"
#include <chrono>


namespace decord {
namespace cuda {
using namespace runtime;

CUThreadedDecoder::CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat)
    : device_id_(device_id), stream_({device_id, false}), device_{}, ctx_{}, parser_{}, decoder_{},
    pkt_queue_{}, frame_queue_{},
    run_(false), frame_count_(0), draining_(false),
    tex_registry_(), nv_time_base_({1, 10000000}), frame_base_({1, 1000000}),
    dec_ctx_(nullptr), bsf_ctx_(nullptr), width_(-1), height_(-1),
    error_status_(false), error_message_() {

    // initialize bitstream filters
    InitBitStreamFilter(codecpar, iformat);

    CHECK_CUDA_CALL(nv::cuInit(0));
    CHECK_CUDA_CALL(nv::cuDeviceGet(&device_, device_id_));

    char device_name[100];
    CHECK_CUDA_CALL(nv::cuDeviceGetName(device_name, 100, device_));
    DLOG(INFO) << "Using device: " << device_name;

    try {
        auto nvml_ret = nv::nvmlInit();
        if (nvml_ret != NVML_SUCCESS) {
            LOG(FATAL) << "nvmlInit returned error " << nvml_ret;
        }
        char nvmod_version_string[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        nvml_ret = nv::nvmlSystemGetDriverVersion(nvmod_version_string,
                                              sizeof(nvmod_version_string));
        if (nvml_ret != NVML_SUCCESS) {
            LOG(FATAL) << "nvmlSystemGetDriverVersion returned error " << nvml_ret;
        }
        auto nvmod_version = std::stof(nvmod_version_string);
        if (nvmod_version < 384.0f) {
            DLOG(INFO) << "Older kernel module version " << nvmod_version
                        << " so using the default stream."
                        << std::endl;
            stream_ = CUStream(device_id_, true);
        } else {
            DLOG(INFO) << "Kernel module version " << nvmod_version
                        << ", so using our own stream.";
        }
    } catch(const std::exception& e) {
        DLOG(INFO) << "Unable to get nvidia kernel module version from NVML, "
                    << "conservatively assuming it is an older version.\n"
                    << "The error was: " << e.what();
        stream_ = CUStream(device_id_, true);
    }

    ctx_ = CUContext(device_);
    if (!ctx_.Initialized()) {
        LOG(FATAL) << "Problem initializing context";
        return;
    }
}

void CUThreadedDecoder::InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat) {
    const char* bsf_name = nullptr;
    if (AV_CODEC_ID_H264 == codecpar->codec_id) {
        // H.264
        bsf_name = "h264_mp4toannexb";
    } else if (AV_CODEC_ID_HEVC == codecpar->codec_id) {
        // HEVC
        bsf_name = "hevc_mp4toannexb";
    } else if (AV_CODEC_ID_MPEG4 == codecpar->codec_id && !strcmp(iformat->name, "avi")) {
        // MPEG4
        bsf_name = "mpeg4_unpack_bframes";
    } else if (AV_CODEC_ID_AV1 == codecpar->codec_id) {
        // AV1.  FFmpeg 7+ removed av1_mp4toannexb: the AV1 decoder consumes
        // the MP4 sample (OBU stream) directly, and NVDEC does too via the
        // null BSF (the AV1C config record travels in extradata).
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(61, 0, 0)
        bsf_name = "av1_mp4toannexb";
#else
        bsf_name = "null";
#endif
    } else {
        bsf_name = "null";
    }

    auto bsf = av_bsf_get_by_name(bsf_name);
    CHECK(bsf) << "Error finding bitstream filter: " << bsf_name;

    AVBSFContext* bsf_ctx = nullptr;
    CHECK_GE(av_bsf_alloc(bsf, &bsf_ctx), 0) << "Error allocating bit stream filter context.";
    CHECK_GE(avcodec_parameters_copy(bsf_ctx->par_in, codecpar), 0) << "Error setting BSF parameters.";
    CHECK_GE(av_bsf_init(bsf_ctx), 0) << "Error init BSF";
    CHECK_GE(avcodec_parameters_copy(codecpar, bsf_ctx->par_out), 0) << "Error copy bsf output to codecpar";
    bsf_ctx_.reset(bsf_ctx);
}

void CUThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation, int output_format) {
    // GPU (NVDEC) path always outputs RGB via improc; gray is CPU-only.
    (void)output_format;
    CHECK(dec_ctx);
    width_ = width;
    height_ = height;
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    parser_ = CUVideoParser(dec_ctx->codec_id, this, kMaxOutputSurfaces, dec_ctx->extradata,
                            dec_ctx->extradata_size);
    if (!parser_.Initialized()) {
        LOG(FATAL) << "Problem creating video parser";
        return;
    }
    if (running) {
        Start();
    }
}

void CUThreadedDecoder::Start() {
    CheckErrorStatus();
    if (run_.load()) return;

    pkt_queue_.reset(new PacketQueue());
    frame_queue_.reset(new FrameQueue());
    // buffer_queue_.reset(new BufferQueue());
    reorder_queue_.reset(new ReorderQueue());
    //frame_order_.reset(new FrameOrderQueue());
    avcodec_flush_buffers(dec_ctx_.get());
    // Recreate the NVDEC decoder alongside the parser. Surfaces left in
    // flight by the previous session (pending decode/display) would
    // otherwise keep firing display callbacks that consume frame_queue_
    // buffers paired 1:1 with newly pushed packets, starving later frames
    // and deadlocking PushNext's backpressure loop.
    decoder_ = CUVideoDecoderImpl();
    parser_ = CUVideoParser(dec_ctx_->codec_id, this, kMaxOutputSurfaces, dec_ctx_->extradata,
                            dec_ctx_->extradata_size);
    if (!parser_.Initialized()) {
        LOG(FATAL) << "Problem creating video parser";
        return;
    }
    // Push() uses cuCtxSetCurrent (no stack); the main-thread current setting
    // does not affect the decode thread (current is thread-local), no pop needed.
    run_.store(true);
    // launch worker threads
    auto launcher_t = std::thread{&CUThreadedDecoder::LaunchThread, this};
    std::swap(launcher_t_, launcher_t);
}

void CUThreadedDecoder::Stop() {
    // a stopped decoder is no longer draining; otherwise a leftover
    // draining_ flag would swallow the next EOF flush marker
    draining_.store(false);
    if (run_.load()) {
        pkt_queue_->SignalForKill();
        run_.store(false);
        frame_queue_->SignalForKill();
        reorder_queue_->SignalForKill();
    }
    pkt_room_cv_.notify_all();  // wake any Push() blocked on backpressure
    if (launcher_t_.joinable()) {
        launcher_t_.join();
    }
}

void CUThreadedDecoder::Clear() {
    Stop();
    frame_count_.store(0);
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      discard_pts_.clear();
    }
    error_status_.store(false);
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_.clear();
    }
}

CUThreadedDecoder::~CUThreadedDecoder() {
    Clear();
}

int CUDAAPI CUThreadedDecoder::HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    // LOG(INFO) << "HandlePictureSequence, thread id: " << std::this_thread::get_id();;
    return decoder->HandlePictureSequence_(format);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDecode(void* user_data,
                                            CUVIDPICPARAMS* pic_params) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    // LOG(INFO) << "HandlePictureDecode, thread id: " << std::this_thread::get_id();;
    return decoder->HandlePictureDecode_(pic_params);
}

int CUDAAPI CUThreadedDecoder::HandlePictureDisplay(void* user_data,
                                             CUVIDPARSERDISPINFO* disp_info) {
    auto decoder = reinterpret_cast<CUThreadedDecoder*>(user_data);
    // LOG(INFO) << "HandlePictureDisplay, thread id: " << std::this_thread::get_id();;
    return decoder->HandlePictureDisplay_(disp_info);
}

int CUThreadedDecoder::HandlePictureSequence_(CUVIDEOFORMAT* format) {
    // width_ = format->coded_width;
    // height_ = format->coded_height;
    frame_base_ = {static_cast<int>(format->frame_rate.denominator),
                   static_cast<int>(format->frame_rate.numerator)};
    return decoder_.Initialize(format);
}

int CUThreadedDecoder::HandlePictureDecode_(CUVIDPICPARAMS* pic_params) {
    if (!run_.load()) return 0;
    CHECK(decoder_.Initialized());
    // CHECK_GE(pic_params->CurrPicIdx, 0);
    // CHECK_LT(pic_params->CurrPicIdx, permits_.size());
    // auto permit_queue = permits_[pic_params->CurrPicIdx];
    // int tmp;
    // while (permit_queue->Size() < 1) continue;
    // int ret = permit_queue->Pop(&tmp);
    if (!CHECK_CUDA_CALL(nv::cuvidDecodePicture(decoder_, pic_params))) {
        LOG(FATAL) << "Failed to launch cuvidDecodePicture";
        return 0;
    }
    // decoded_cnt_++;
    return 1;
}

int CUThreadedDecoder::HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info) {
    if (!run_.load()) return 0;
    // push to converter
    // LOG(INFO) << "frame in use occupy: " << disp_info->picture_index;
    // frame_in_use_[disp_info->picture_index] = 1;
    // buffer_queue_->Push(disp_info);
    // finished, send clear msg to allow next decoding
    // LOG(INFO) << "finished, send clear msg to allow next decoding";
    NDArray arr;
    frame_queue_->Pop(&arr);
    if (!arr.defined()) {
        return 0;
    }
    arr.pts = disp_info->timestamp;
    bool skip = false;
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      skip = discard_pts_.find(disp_info->timestamp) != discard_pts_.end();
    }
    if (skip) {
        // skip frame processing
        reorder_queue_->Push(arr);
        return 1;
    }

    uint8_t* dst_ptr = static_cast<uint8_t*>(arr.data_->dl_tensor.data);
    auto frame = CUMappedFrame(disp_info, decoder_, stream_);
    // int64_t frame_pts = static_cast<int64_t>(frame.disp_info->timestamp);
    auto input_width = decoder_.Width();
    auto input_height = decoder_.Height();
    auto& textures = tex_registry_.GetTexture(frame.get_ptr(),
                                            frame.get_pitch(),
                                            input_width,
                                            input_height,
                                            ScaleMethod_Linear,
                                            ChromaUpMethod_Linear,
                                            decoder_.BitDepth());
    ProcessFrame(textures.chroma, textures.luma, dst_ptr, stream_, input_width, input_height, width_, height_, decoder_.BitDepth());
    if (!CHECK_CUDA_CALL(cudaStreamSynchronize(stream_))) {
        LOG(FATAL) << "Error synchronize cuda stream";
        return 0;
    }
    reorder_queue_->Push(arr);
    return 1;
}

void CUThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.insert(dts.begin(), dts.end());
}

void CUThreadedDecoder::ClearDiscardPTS() {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.clear();
}

void CUThreadedDecoder::Push(AVPacketPtr pkt, NDArray buf) {
    CHECK(run_.load());
    if (!pkt && draining_.load()) {
        // already draining: the EOF marker was enqueued on the first null
        // push, but every pushed buffer must still reach frame_queue_ so
        // display callbacks for in-flight frames can always pop one and
        // never block (a blocked display callback stalls the whole NVDEC
        // flush and silently drops frames)
        frame_queue_->Push(buf);
        ++frame_count_;
        return;
    }
    if (!pkt) draining_.store(true);

    {
        // Backpressure: wait until the queue drains below the surface
        // limit. The decoder thread notifies after every Pop; the old
        // 1ns busy-poll burned a full core while the decoder saturated.
        std::unique_lock<std::mutex> lock(pkt_room_mutex_);
        pkt_room_cv_.wait(lock, [this]() {
            return !run_.load() || pkt_queue_->Size() <= kMaxOutputSurfaces;
        });
    }

    pkt_queue_->Push(pkt);
    frame_queue_->Push(buf);  // push memory buffer
    ++frame_count_;
}

bool CUThreadedDecoder::Pop(NDArray *frame) {
    CheckErrorStatus();
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    if (reorder_queue_->Size() < 1) {
        return false;
    }
    int ret = reorder_queue_->Pop(frame);
    CheckErrorStatus();
    if (!ret) return false;
    --frame_count_;
    return true;
}

void CUThreadedDecoder::LaunchThread() {
  try {
      LaunchThreadImpl();
  } catch (dmlc::Error error) {
      RecordInternalError(error.what());
      run_.store(false);
      frame_queue_->SignalForKill(); // Unblock all consumers
  } catch (std::exception& error) {
      RecordInternalError(error.what());
      run_.store(false);
      frame_queue_->SignalForKill(); // Unblock all consumers
  }
}

void CUThreadedDecoder::LaunchThreadImpl() {
    ctx_.Push();
    // LOG(INFO) << "LaunchThread, thread id: " << std::this_thread::get_id();
    while (run_.load()) {
        bool ret;
        AVPacketPtr avpkt = nullptr;
        ret = pkt_queue_->Pop(&avpkt);
        if (!ret) return;
        // a slot freed: wake any Push() waiting on backpressure
        pkt_room_cv_.notify_one();

        if (avpkt && avpkt->size) {
            // Strip side data before BSF: the hardware decoder only needs raw
            // bitstream data.  FFmpeg 8.x may attach side data (HDR metadata,
            // display matrix, etc.) to every packet; clearing it avoids
            // unnecessary allocation/copy overhead in the BSF hot path.
            av_packet_free_side_data(avpkt.get());
            // bitstream filter raw packet
            AVPacketPtr filtered_avpkt = ffmpeg::AVPacketPool::Get()->Acquire();
            if (filtered_avpkt->data) {
                av_packet_unref(filtered_avpkt.get());
            }
            CHECK(av_bsf_send_packet(bsf_ctx_.get(), avpkt.get()) == 0) << "Error sending BSF packet";
            int bsf_ret;
            while ((bsf_ret = av_bsf_receive_packet(bsf_ctx_.get(), filtered_avpkt.get())) == 0) {
                CUVIDSOURCEDATAPACKET cupkt = {0};
                cupkt.payload_size = filtered_avpkt->size;
                cupkt.payload = filtered_avpkt->data;
                if (filtered_avpkt->pts != AV_NOPTS_VALUE) {
                    cupkt.flags = CUVID_PKT_TIMESTAMP;
                    cupkt.timestamp = filtered_avpkt->pts;
                }

                if (!CHECK_CUDA_CALL(nv::cuvidParseVideoData(parser_, &cupkt))) {
                    LOG(FATAL) << "Problem decoding packet";
                }
            }
        } else {
            CUVIDSOURCEDATAPACKET cupkt = {0};
            cupkt.flags = CUVID_PKT_ENDOFSTREAM;
            if (!CHECK_CUDA_CALL(nv::cuvidParseVideoData(parser_, &cupkt))) {
                LOG(FATAL) << "Problem decoding packet";
            }
            // cuvidParseVideoData callbacks are synchronous, so after
            // ENDOFSTREAM returns every remaining frame has been displayed
            // and pushed to reorder_queue_. Signal that draining is done the
            // same way the CPU decoder does, so NextFrameImpl can fall back
            // to cached frames / rewind recovery instead of spinning.
            for (int i = 0; i < ThreadedDecoderInterface::kDrainMarkerCount; ++i) {
                reorder_queue_->Push(NDArray::Empty({1}, kInt64, kCPU));
            }
        }
    }
}

void CUThreadedDecoder::CheckErrorStatus() {
    if (error_status_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        LOG(FATAL) << error_message_;
    }
}

void CUThreadedDecoder::RecordInternalError(std::string message) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = message;
    }
    error_status_.store(true);
}

}  // namespace cuda
}  // namespace decord
