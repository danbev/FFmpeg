/*
 * Copyright (c) 2025 ggml-org
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <parakeet.h>
#include <ggml-backend.h>

#include "libavutil/avutil.h"
#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/audio.h"
#include "libavutil/mem.h"
#include "libavutil/avstring.h"
#include "libavutil/internal.h"
#include "libavformat/avio.h"
#include "libavutil/thread.h"

#include "formats.h"

typedef struct ParakeetContext {
    const AVClass *class;
    char *model_path;
    bool use_gpu;
    int gpu_device;

    char *destination;
    char *format;

    struct parakeet_context     *ctx_pkt;
    struct parakeet_state       *state_pkt;
    struct parakeet_full_params  full_params;

    float  *samples_buf;
    int     n_samples;
    int     samples_capacity;

    int eof;
    int64_t next_pts;

    AVIOContext *avio_context;
    int index;
} ParakeetContext;

static void cb_log(enum ggml_log_level level, const char *text, void *user_data)
{
    AVFilterContext *ctx = user_data;
    int av_log_level = AV_LOG_DEBUG;
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        av_log_level = AV_LOG_ERROR;
        break;
    case GGML_LOG_LEVEL_WARN:
        av_log_level = AV_LOG_WARNING;
        break;
    }
    av_log(ctx, av_log_level, "%s", text);
}

static int init(AVFilterContext *ctx)
{
    ParakeetContext *pctx = ctx->priv;

    static AVOnce init_static_once = AV_ONCE_INIT;
    ff_thread_once(&init_static_once, ggml_backend_load_all);

    parakeet_log_set(cb_log, ctx);

    if (!pctx->model_path) {
        av_log(ctx, AV_LOG_ERROR, "No parakeet model path specified. Use the 'model' option.\n");
        return AVERROR(EINVAL);
    }

    if (av_strcasecmp(pctx->format, "text") &&
        av_strcasecmp(pctx->format, "srt") &&
        av_strcasecmp(pctx->format, "json")) {
        av_log(ctx, AV_LOG_ERROR, "Invalid format '%s'. Valid formats are: text, srt, json.\n", pctx->format);
        return AVERROR(EINVAL);
    }

    struct parakeet_context_params params = parakeet_context_default_params();
    params.use_gpu    = pctx->use_gpu;
    params.gpu_device = pctx->gpu_device;

    pctx->ctx_pkt = parakeet_init_from_file_with_params_no_state(pctx->model_path, params);
    if (pctx->ctx_pkt == NULL) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize parakeet context from model: %s\n", pctx->model_path);
        return AVERROR(EIO);
    }

    pctx->state_pkt = parakeet_init_state(pctx->ctx_pkt);
    if (pctx->state_pkt == NULL) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize parakeet state\n");
        parakeet_free(pctx->ctx_pkt);
        return AVERROR(ENOMEM);
    }

    pctx->full_params            = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    pctx->full_params.n_threads  = ff_filter_get_nb_threads(ctx);
    pctx->full_params.no_context = false;

    pctx->next_pts = AV_NOPTS_VALUE;

    if (pctx->destination && strcmp("", pctx->destination)) {
        const char *dst = pctx->destination;
        if (!strcmp("-", dst))
            dst = "pipe:1";
        int ret = avio_open(&pctx->avio_context, dst, AVIO_FLAG_WRITE);

        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not open %s: %s\n", pctx->destination, av_err2str(ret));
            return ret;
        }

        pctx->avio_context->direct = AVIO_FLAG_DIRECT;
    }

    av_log(ctx, AV_LOG_INFO, "Parakeet filter initialized: model: %s\n", pctx->model_path);

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    ParakeetContext *pctx = ctx->priv;

    if (pctx->state_pkt) {
        parakeet_free_state(pctx->state_pkt);
        pctx->state_pkt = NULL;
    }

    if (pctx->ctx_pkt) {
        parakeet_free(pctx->ctx_pkt);
        pctx->ctx_pkt = NULL;
    }

    if (pctx->avio_context) {
        avio_closep(&pctx->avio_context);
    }

    av_freep(&pctx->samples_buf);
    pctx->n_samples       = 0;
    pctx->samples_capacity = 0;
}

static void collect_new_segments(AVFilterContext *ctx, int segments_before, char **segments_text)
{
    ParakeetContext *pctx = ctx->priv;
    const int total_segments = parakeet_full_n_segments_from_state(pctx->state_pkt);

    const int new_segments = total_segments - segments_before;

    av_log(ctx, AV_LOG_DEBUG, "Generated %d new segments\n", new_segments);

    for (int i = segments_before; i < total_segments; ++i) {
        const char *text = parakeet_full_get_segment_text_from_state(pctx->state_pkt, i);
        const char *trimmed_text;

        if (!text || text[0] == '\0')
            continue;

        trimmed_text = text;
        while (av_isspace(trimmed_text[0])) {
            trimmed_text++;
        }
        if (trimmed_text[0] == '\0') {
            continue;
        }

        const int64_t t0_cs = parakeet_full_get_segment_t0_from_state(pctx->state_pkt, i);
        const int64_t t1_cs = parakeet_full_get_segment_t1_from_state(pctx->state_pkt, i);
        const int64_t t0_ms = t0_cs * 10;
        const int64_t t1_ms = t1_cs * 10;

        av_log(ctx, AV_LOG_DEBUG, "  [%" PRId64 "-%" PRId64 "]: \"%s\"\n", t0_ms, t1_ms, text);

        if (*segments_text) {
            char *new_text = av_asprintf("%s%s", *segments_text, text);
            av_freep(segments_text);
            *segments_text = new_text;
        } else {
            *segments_text = av_strdup(text);
        }

        if (pctx->avio_context) {
            char *buf = NULL;

            if (!av_strcasecmp(pctx->format, "srt")) {
                buf = av_asprintf(
                    "%d\n%02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64 " --> %02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64 "\n%s\n\n",
                    pctx->index, t0_ms / 3600000,
                    (t0_ms / 60000) % 60, (t0_ms / 1000) % 60,
                    t0_ms % 1000, t1_ms / 3600000, (t1_ms / 60000) % 60,
                    (t1_ms / 1000) % 60, t1_ms % 1000, trimmed_text);
                pctx->index++;
            } else if (!av_strcasecmp(pctx->format, "json")) {
                buf = av_asprintf("{\"start\":%" PRId64 ",\"end\":%" PRId64 ",\"text\":\"%s\"}\n", t0_ms, t1_ms, trimmed_text);
            } else {
                buf = av_asprintf("%s", text);
            }

            if (buf) {
                avio_write(pctx->avio_context, buf, strlen(buf));
                av_freep(&buf);
            }
        }
    }

}

static int run_transcription(AVFilterContext *ctx,
                             const float *samples,
                             int nb_samples,
                             int flush,
                             char **segments_text)
{
    ParakeetContext *pctx = ctx->priv;
    int segments_before;
    int ret;

    if (!pctx->ctx_pkt || !pctx->state_pkt) {
        return 0;
    }

    if (!flush) {
        // Buffer samples until EOF.
        const int needed = pctx->n_samples + nb_samples;
        if (needed > pctx->samples_capacity) {
            float *new_buf = av_realloc(pctx->samples_buf, needed * sizeof(float));
            if (!new_buf)
                return AVERROR(ENOMEM);
            pctx->samples_buf      = new_buf;
            pctx->samples_capacity = needed;
        }
        memcpy(pctx->samples_buf + pctx->n_samples, samples, nb_samples * sizeof(float));
        pctx->n_samples += nb_samples;
        return 0;
    }

    if (pctx->n_samples == 0)
        return 0;

    av_log(ctx, AV_LOG_INFO, "Transcribing %d samples (%.2f seconds)\n",
           pctx->n_samples, (float) pctx->n_samples / PARAKEET_SAMPLE_RATE);

    segments_before = parakeet_full_n_segments_from_state(pctx->state_pkt);

    ret = parakeet_full_with_state(pctx->ctx_pkt, pctx->state_pkt, pctx->full_params,
                                   pctx->samples_buf, pctx->n_samples);
    pctx->n_samples = 0;

    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "parakeet_full_with_state failed: %d\n", ret);
        return AVERROR_EXTERNAL;
    }

    collect_new_segments(ctx, segments_before, segments_text);
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    ParakeetContext *pctx = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    const int samples = frame->nb_samples;
    const float *input_data = (const float *) frame->data[0];
    const float duration = (float) samples / PARAKEET_SAMPLE_RATE;
    char *segments_text = NULL;
    int ret;

    ret = run_transcription(ctx, input_data, samples, 0, &segments_text);
    if (ret < 0) {
        av_freep(&segments_text);
        av_frame_free(&frame);
        return ret;
    }

    if (segments_text && segments_text[0] != '\0') {
        av_dict_set(&frame->metadata, "lavfi.parakeet.text", segments_text, 0);
        char *duration_text = av_asprintf("%f", duration);
        av_dict_set(&frame->metadata, "lavfi.parakeet.duration", duration_text, AV_DICT_DONT_STRDUP_VAL);
    }
    av_freep(&segments_text);

    pctx->next_pts = frame->pts + av_rescale_q(samples, (AVRational) {
                                               1, inlink->sample_rate}
                                               , inlink->time_base);
    return ff_filter_frame(outlink, frame);
}

static int push_last_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ParakeetContext *pctx = ctx->priv;
    AVFrame *frame;
    char *segments_text = NULL;
    int ret;

    if (ctx->is_disabled) {
        return 0;
    }

    ret = run_transcription(ctx, NULL, 0, 1, &segments_text);
    if (ret < 0) {
        av_freep(&segments_text);
        return ret;
    }

    if (!segments_text || segments_text[0] == '\0') {
        av_freep(&segments_text);
        return 0;
    }

    frame = ff_get_audio_buffer(outlink, 1);
    if (!frame) {
        av_freep(&segments_text);
        return AVERROR(ENOMEM);
    }

    av_samples_set_silence(frame->extended_data, 0, 1, frame->ch_layout.nb_channels, frame->format);

    frame->pts = pctx->next_pts;
    if (pctx->next_pts != AV_NOPTS_VALUE)
        pctx->next_pts += av_rescale_q(1, (AVRational) { 1, outlink->sample_rate }, outlink->time_base);

    av_dict_set(&frame->metadata, "lavfi.parakeet.text", segments_text, 0);
    av_dict_set(&frame->metadata, "lavfi.parakeet.duration", av_strdup("0.000000"), AV_DICT_DONT_STRDUP_VAL);
    av_freep(&segments_text);

    return ff_filter_frame(outlink, frame);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    ParakeetContext *pctx = ctx->priv;
    int64_t pts;
    int status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!pctx->eof && ff_inlink_queued_frames(inlink)) {
        AVFrame *frame = NULL;
        int ret;

        // pull an AVFrame from the input queue.
        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0) {
            return ret;
        }
        if (ret > 0) {
            return filter_frame(inlink, frame);
        }
    }

    if (!pctx->eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        pctx->eof = status == AVERROR_EOF;
    }

    if (pctx->eof) {
        int ret = push_last_frame(outlink);
        if (ret < 0)
            return ret;

        ff_outlink_set_status(outlink, AVERROR_EOF, pctx->next_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_NONE };
    AVChannelLayout chlayouts[] = { FF_COUNT2LAYOUT(1), { 0 } };
    const int sample_rates[] = { PARAKEET_SAMPLE_RATE, -1 };
    int ret;

    ret = ff_set_sample_formats_from_list2(ctx, cfg_in, cfg_out, sample_fmts);
    if (ret < 0) {
        return ret;
    }

    ret = ff_set_common_channel_layouts_from_list2(ctx, cfg_in, cfg_out, chlayouts);
    if (ret < 0) {
        return ret;
    }

    return ff_set_common_samplerates_from_list2(ctx, cfg_in, cfg_out, sample_rates);
}

#define OFFSET(x) offsetof(ParakeetContext, x)
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption parakeet_options[] = {
    { "model",       "Path to the parakeet.cpp model file", OFFSET(model_path),  AV_OPT_TYPE_STRING, {.str = NULL}, .flags = FLAGS },
    { "use_gpu",     "Use GPU for processing",              OFFSET(use_gpu),     AV_OPT_TYPE_BOOL,   {.i64 = 1}, 0, 1, .flags = FLAGS },
    { "gpu_device",  "GPU device to use",                   OFFSET(gpu_device),  AV_OPT_TYPE_INT,    {.i64 = 0}, 0, INT_MAX, .flags = FLAGS },
    { "destination", "Output destination",                  OFFSET(destination), AV_OPT_TYPE_STRING, {.str = ""}, .flags = FLAGS },
    { "format",      "Output format (text|srt|json)",       OFFSET(format),      AV_OPT_TYPE_STRING, {.str = "text"}, .flags = FLAGS },
    { NULL }
};

static const AVClass parakeet_class = {
    .class_name = "parakeet",
    .item_name  = av_default_item_name,
    .option     = parakeet_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFFilter ff_af_parakeet = {
    .p.name        = "parakeet",
    .p.description = NULL_IF_CONFIG_SMALL("Transcribe audio using parakeet.cpp."),
    .p.priv_class  = &parakeet_class,
    .p.flags       = AVFILTER_FLAG_METADATA_ONLY,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(ParakeetContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
