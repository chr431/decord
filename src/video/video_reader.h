/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_reader.h
 * \brief FFmpeg video reader, implements VideoReaderInterface
 */

#ifndef DECORD_VIDEO_VIDEO_READER_H_
#define DECORD_VIDEO_VIDEO_READER_H_

#include "threaded_decoder_interface.h"
#include "storage_pool.h"
#include <decord/video_interface.h>

#include <string>
#include <vector>

#include <decord/base.h>
#include <dmlc/concurrency.h>


namespace decord {
using timestamp_t = float;
struct AVFrameTime {
    int64_t pts;          // presentation timestamp, unit is stream time_base
    int64_t dts;          // decoding timestamp, unit is stream time_base
    timestamp_t start;    // real world start timestamp, unit is second
    timestamp_t stop;     // real world stop timestamp, unit is second

    AVFrameTime(int64_t pts=AV_NOPTS_VALUE, int64_t dts=AV_NOPTS_VALUE, timestamp_t start=0, timestamp_t stop=0)
     : pts(pts), dts(dts), start(start), stop(stop) {}
};  // struct AVFrameTime

class VideoReader : public VideoReaderInterface {
    using ThreadedDecoderPtr = std::unique_ptr<ThreadedDecoderInterface>;
    using NDArray = runtime::NDArray;
    public:
        VideoReader(std::string fn, DLDevice ctx, int width=-1, int height=-1,
                    int nb_thread=0, int io_type=kNormal, std::string fault_tol="-1",
                    int output_format = 0);
        /*! \brief Destructor, note that FFMPEG resources has to be managed manually to avoid resource leak */
        ~VideoReader();
        void SetVideoStream(int stream_nb = -1);
        unsigned int QueryStreams() const;
        int64_t GetFrameCount() const;
        int64_t GetCurrentPosition() const;
        NDArray NextFrame();
        /*!
         * \brief Grab the next frame and return only the ROI rectangle.
         * \param x1,y1,x2,y2 Half-open ROI [x1,x2) x [y1,y2).  On the GPU path
         *        only the ROI rectangle is copied to host memory (the caller
         *        would otherwise receive the full frame and crop it, wasting
         *        a full-frame D2H copy).  CPU builds and invalid ROIs fall
         *        back to the full frame (caller crops).
         */
        NDArray NextFrameRoi(int x1, int y1, int x2, int y2);
        /*!
         * \brief 固定 ROI 输出（ROI-first 解码管线）。
         * \param x1,y1,x2,y2 半开 ROI [x1,x2) x [y1,y2)（全帧坐标）。
         *        无效矩形（w<=0/h<=0/越界）清除 ROI 回退全帧输出。
         * 须在任何帧解码前调用（解码器 filter 图 / GPU 转换器 / 输出池
         * 在此按 ROI 重建）。此后 NextFrameRoi/GetBatch 的每帧 ROI 应与
         * 之完全一致（python 层强制）；一致时 CropRoi 直通（无每帧拷贝）。
         */
        void SetRoi(int x1, int y1, int x2, int y2);
        /*!
         * \brief Grab a batch of frames; an optional ROI rectangle crops
         *        every frame to [x1,x2) x [y1,y2) before writing into the
         *        batch buffer (half-open, same semantics as NextFrameRoi).
         *        ROI < 0 (any coordinate) returns full frames, keeping the
         *        historical behaviour.  Batch shape becomes
         *        [N, y2-y1, x2-x1, 3] when a valid ROI is given.
         */
        NDArray GetBatch(std::vector<int64_t> indices, NDArray buf,
                         int x1 = -1, int y1 = -1, int x2 = -1, int y2 = -1);
        void SkipFrames(int64_t num = 1);
        /*!
         * \brief Seek to a frame position by PTS.
         *
         * force_backward: always use AVSEEK_FLAG_BACKWARD (land at the
         * keyframe AT OR BEFORE the target PTS).  Without it, seeking to a
         * timestamp equal to a keyframe's own PTS can land one keyframe
         * late (sparse-GOP VFR videos; see SeekAccurate).
         */
        bool Seek(int64_t pos);
        bool Seek(int64_t pos, bool force_backward);
        bool SeekAccurate(int64_t pos);
        NDArray GetKeyIndices();
        NDArray GetFramePTS() const;
        double GetAverageFPS() const;
        double GetRotation() const;
        std::string GetCodec() const;
        /*! \brief 流 color_range：0 = limited/tv, 1 = full/pc（Y 展开语义）。 */
        int GetColorRange() const { return color_range_; }
    protected:
        friend class VideoLoader;
        std::vector<int64_t> GetKeyIndicesVector() const;
    private:
        void IndexKeyframes();
        /*! \brief Try to load the frame/keyframe index from the on-disk
         *  index cache (keyed by file size + mtime).  Returns true on a
         *  fresh hit; the caller falls back to IndexKeyframes(). */
        bool LoadCachedIndex();
        /*! \brief Persist the freshly built index to the on-disk cache. */
        void SaveCachedIndex() const;
        /*! \brief Cache file path for this video (empty if uncacheable). */
        std::string IndexCachePath() const;
        void PushNext();
        int64_t LocateKeyframe(int64_t pos);
        void SkipFramesImpl(int64_t num = 1);
        bool CheckKeyFrame();
        NDArray NextFrameImpl();
        int64_t FrameToPTS(int64_t pos);
        std::vector<int64_t> FramesToPTS(const std::vector<int64_t>& positions);
        void CacheFrame(NDArray frame);
        bool FetchCachedFrame(NDArray &frame, int64_t pos);
        /*! \brief Row-stride copy of a frame's ROI rectangle (CPU memcpy or
         *  GPU cudaMemcpy2D).  Shared by NextFrameRoi and GetBatch(roi). */
        NDArray CropRoi(NDArray frame, int x1, int y1, int x2, int y2);
        /*! \brief ROI crop for packed NV12 frames (Y rows + interleaved UV). */
        NDArray CropRoiYuv420(NDArray frame, int x1, int y1, int x2, int y2);

        DLDevice ctx_;
        std::vector<int64_t> key_indices_;
        std::map<int64_t, int64_t> pts_frame_map_;
        NDArray tmp_key_frame_;
        bool overrun_;
        /*! \brief a lookup table for per frame pts/dts */
        std::vector<AVFrameTime> frame_ts_;
        /*! \brief Video Streams Codecs in original videos */
        std::vector<const AVCodec*> codecs_;
        /*! \brief Currently active video stream index */
        int actv_stm_idx_;
        /*! \brief AV format context holder */
        ffmpeg::AVFormatContextPtr fmt_ctx_;
        ThreadedDecoderPtr decoder_;
        int64_t curr_frame_;  // current frame location
        int64_t nb_thread_decoding_;  // number of threads for decoding
        int width_;   // output video width
        int height_;  // output video height
        int output_format_;  // 0 = RGB24, 1 = GRAY8, 2 = YUV420/NV12
        int color_range_ = 0;  // 0 = limited/tv, 1 = full/pc（Y 展开语义）
        // RGB/gray use 3D frames (h,w,c); yuv420 uses a packed 2D frame
        // (h+ceil(h/2), w) — Y rows first, then interleaved U/V rows.
        int OutputChannels() const { return output_format_ == 1 ? 1 : 3; }
        bool IsYuv420() const { return output_format_ == 2; }
        int64_t FrameRows(int64_t h) const {
            return IsYuv420() ? h + (h + 1) / 2 : h;
        }
        std::vector<int64_t> FrameShape(int64_t h, int64_t w) const {
            if (IsYuv420()) return {FrameRows(h), w};
            return {h, w, OutputChannels()};
        }
        // ── ROI-first 状态（SetRoi）──
        int roi_x1_ = 0, roi_y1_ = 0, roi_x2_ = -1, roi_y2_ = -1;  // 半开
        bool has_roi_ = false;
        int roi_w_ = 0, roi_h_ = 0;
        // CPU filter 层实际裁剪矩形（yuv420p 上 FFmpeg crop 要求偶数宽高，
        // 奇数尺寸产全黑/崩溃 —— 此处按偶数扩充，CropRoi 顶部对齐精裁）
        int roi_fx1_ = 0, roi_fy1_ = 0, roi_fw_ = 0, roi_fh_ = 0;
        bool eof_;  // end of file indicator
        /*! \brief decoder queue in-flight depth accounting.  NextFrameImpl
         *  keeps pkts_pushed_ - frames_popped_ near kPrefetchDepth so the
         *  decoder threads / GPU surfaces stay busy instead of being
         *  latency-bound on one packet at a time.  Reset on Seek (the
         *  decoder queue is cleared there). */
        int64_t pkts_pushed_ = 0;
        int64_t frames_popped_ = 0;
        NDArrayPool ndarray_pool_;
        std::unique_ptr<ffmpeg::AVIOBytesContext> io_ctx_;  // avio context for raw memory access
        std::string filename_;  // file name if from file directly, can be empty if from bytes
        NDArray cached_frame_;  // last valid frame, for error tolerance
        bool use_cached_frame_;  // switch to enable frame recovery if failed to decode
        std::unordered_set<int64_t> failed_idx_;  // idx of failed frames(recovered from other frames)
        int64_t fault_tol_thresh_;  // fault tolerance threshold, raise if recovered frames retrieved exceeds thresh
        bool fault_warn_emit_;  // whether a fault warning has been emitted
};  // class VideoReader
}  // namespace decord
#endif  // DECORD_VIDEO_VIDEO_READER_H_
