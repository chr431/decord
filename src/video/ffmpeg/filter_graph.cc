/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file filter_graph.cc
 * \brief FFmpeg Filter Graph Impl
 */

#include "filter_graph.h"
#include <thread>

#include <dmlc/logging.h>

namespace decord {
namespace ffmpeg {

FFMPEGFilterGraph::FFMPEGFilterGraph(std::string filters_descr, AVCodecContext *dec_ctx, int output_format)
    : buffersink_ctx_(nullptr), buffersrc_ctx_(nullptr), filter_graph_(nullptr), count_(0) {
    Init(filters_descr, dec_ctx, output_format);
}

FFMPEGFilterGraph::~FFMPEGFilterGraph() {
    // avfilter_free(buffersink_ctx_);
    // avfilter_free(buffersrc_ctx_);
    // avfilter_graph_free(&filter_graph_);
}

void FFMPEGFilterGraph::Init(std::string filters_descr, AVCodecContext *dec_ctx, int output_format) {
    char args[512];
    #if LIBAVFILTER_VERSION_INT < AV_VERSION_INT(7,14,100)
    avfilter_register_all();
    #endif
    const AVFilter *buffersrc  = avfilter_get_by_name("buffer");
	const AVFilter *buffersink = avfilter_get_by_name("buffersink");
    if (!buffersink) {
        buffersink = avfilter_get_by_name("ffbuffersink");
    }
    CHECK(buffersrc) << "Error: no buffersrc";
    CHECK(buffersink) << "Error: no buffersink";
    AVFilterInOut *outputs = avfilter_inout_alloc();
	AVFilterInOut *inputs  = avfilter_inout_alloc();
	enum AVPixelFormat pix_fmts[] = { output_format ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE };
	// AVBufferSinkParams *buffersink_params;

	filter_graph_.reset(avfilter_graph_alloc());
	/* nb_threads: 0 = auto.  Auto is faster for standalone decode (878fps
	 * vs 1-thread), but in a decode/OCR pipeline it competes with the ONNX
	 * inference threads for cores and ends up slower overall (measured
	 * 133fps -> 87fps in-pipeline).  Single-threaded scale is the safer
	 * choice for the pipeline case (dmlc/decord PR #63 rationale).
	 *
	 * 2026-08: the old in-pipeline measurement predates the pipelined
	 * send/receive loop; with prefetch + non-blocking receive the scale
	 * (sws_scale YUV->RGB) became the serial per-frame cost (~0.8ms @1080p
	 * on one thread).  Default scales with the logical CPU count
	 * (clamp(hw/16, 1, 4)): 16-core machines stay at 1 slice thread (the
	 * decode/OCR pipeline optimum; 4 threads measured slower in-pipeline),
	 * machines with spare cores get a little sws parallelism.
	 * DECORD_FILTER_THREADS overrides (0 = auto = all cores; 4 = four
	 * slice threads; 1 = historical single-thread). */
	int filter_threads = []() {
		unsigned hw = std::thread::hardware_concurrency();
		if (hw == 0) hw = 8;
		int n = static_cast<int>(hw) / 16;
		return std::max(1, std::min(n, 4));
	}();
	const char *env_ft = getenv("DECORD_FILTER_THREADS");
	if (env_ft != nullptr && *env_ft != '\0') {
		try {
			filter_threads = std::stoi(env_ft);
		} catch (const std::exception &) {
			filter_threads = 4;
		}
	}
	filter_graph_->nb_threads = filter_threads;
    /* buffer video source: the decoded frames from the decoder will be inserted here. */
	std::snprintf(args, sizeof(args),
            "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
            dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
            dec_ctx->time_base.num, dec_ctx->time_base.den,
            dec_ctx->sample_aspect_ratio.num, dec_ctx->sample_aspect_ratio.den);
    // std::snprintf(args, sizeof(args),
    //         "video_size=%dx%d:pix_fmt=%d",
    //         dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt);

    // LOG(INFO) << "filter args: " << args;

    // AVFilterContext *buffersrc_ctx;
    // AVFilterContext *buffersink_ctx;
    CHECK_GE(avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in",
		args, NULL, filter_graph_.get()), 0) << "Cannot create buffer source";

    // LOG(INFO) << "create filter src";

    /* buffer video sink: to terminate the filter chain. */
	// buffersink_params = av_buffersink_params_alloc();
	// buffersink_params->pixel_fmts = pix_fmts;
	CHECK_GE(avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out",
		NULL, NULL, filter_graph_.get()), 0) << "Cannot create buffer sink";
	// av_free(buffersink_params);
    // LOG(INFO) << "create filter sink";
    // CHECK_GE(av_opt_set_bin(buffersink_ctx_, "pix_fmts", (uint8_t *)&pix_fmts, sizeof(AV_PIX_FMT_RGB24), AV_OPT_SEARCH_CHILDREN), 0) << "Set bin error";
#if LIBAVFILTER_VERSION_INT >= AV_VERSION_INT(10, 0, 0)
    // FFmpeg 7+: pix_fmts on buffersink became an AV_OPT_TYPE_PIXEL_FMT
    // option and can no longer be set as an int list (av_opt_set_int_list
    // fails with AVERROR_OPTION_NOT_FOUND).  The format=... filter at the
    // end of the chain controls the output there; the buffersink follows it
    // without an explicit constraint.
    (void)pix_fmts;
#else
    CHECK_GE(av_opt_set_int_list(buffersink_ctx_, "pix_fmts", pix_fmts,
                                 AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN), 0)
        << "Set output pixel format error.";
#endif

    // LOG(INFO) << "create filter set opt";
    /* Endpoints for the filter graph. */
	outputs->name       = av_strdup("in");
	outputs->filter_ctx = buffersrc_ctx_;
	outputs->pad_idx    = 0;
	outputs->next       = NULL;

	inputs->name       = av_strdup("out");
	inputs->filter_ctx = buffersink_ctx_;
	inputs->pad_idx    = 0;
	inputs->next       = NULL;

    /* Parse filter description */
    CHECK_GE(avfilter_graph_parse_ptr(filter_graph_.get(), filters_descr.c_str(),
		&inputs, &outputs, NULL), 0) << "Failed to parse filters description.";

    /* Config filter graph */
    CHECK_GE(avfilter_graph_config(filter_graph_.get(), NULL), 0) << "Failed to config filter graph";

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
}

void FFMPEGFilterGraph::Push(AVFrame *frame) {
    // push decoded frame into filter graph
    CHECK_GE(av_buffersrc_add_frame_flags(buffersrc_ctx_, frame, AV_BUFFERSRC_FLAG_KEEP_REF), 0)
        << "Error while feeding the filter graph";
    ++count_;
}

bool FFMPEGFilterGraph::Pop(AVFrame **frame) {
    if (!count_.load()) {
        // LOG(INFO) << "No count in filter graph.";
        return false;
    }
    if (!*frame) *frame = av_frame_alloc();
    int ret = av_buffersink_get_frame(buffersink_ctx_, *frame);
    if (ret < 0) LOG(INFO) << "buffersink get frame failed" << AVERROR(ret);
    return ret >= 0;
}

}  // namespace ffmpeg
}  // namespace decord
