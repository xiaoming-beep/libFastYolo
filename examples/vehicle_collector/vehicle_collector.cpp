// 车辆按跟踪摘拍器：RTSP 硬解 → 全幅(或可选 ROI) → RKNN 检测+ByteTrack → 每辆新车存一张
// 职责：为微调采集"不同车辆"样本；同一 track 只落一张图，避免同车连拍重复
// 结构：GStreamer 管线（mppvideodec 硬解 + appsink NV12）→ NV12 转 BGR →
//       libFastYolo track()（检测+跟踪）→ 新 track 落图（原图，可选检测框图）→ 达标退出
// 依赖：OpenCV 4.12(GStreamer) + libFastYolo.so + libyolo_rknn.so + librknnrt.so

#include <fast_yolo/yolo.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_run{true};

void on_sigint(int) { g_run = false; }

// 参数（命令行覆盖默认值）
struct Params {
    std::string model = "/data/01/works/video-projects/models/yolov8n_forkopt_rk3576.rknn";
    std::string url;                                    // RTSP 地址
    std::string tag = "cam";                            // 文件名前缀（一般传相机名）
    std::string out_dir = "/data/01/works/video-projects/raw-frames";
    int roi[4] = {0, 0, 0, 0};                          // 全 0 = 不裁剪用全幅
    bool has_roi = false;
    int interval_ms = 200;                              // 处理间隔（5Hz，配合跟踪关联）
    int max_saves = 20;                                 // 存够多少张就退出
    int duration_s = 1500;                              // 最长运行时长（秒）
    float score = 0.25f;
    float iou = 0.45f;
    bool save_det = false;                              // 是否同时存检测框图
    std::set<int> classes = {2, 3, 5, 7};               // COCO: 车/摩托/巴士/卡车
};

void print_usage(const char *prog) {
    printf("用法: %s --url <rtsp> [--tag 相机名] [选项]\n"
           "  --model   <rknn>      默认 %s\n"
           "  --tag     <名称>       文件名前缀, 默认 cam\n"
           "  --out     <目录>       默认 %s\n"
           "  --roi     x1,y1,x2,y2  可选; 缺省不裁剪用全幅\n"
           "  --interval <毫秒>      处理间隔, 默认 500\n"
           "  --max-saves <数量>     每相机最多存几张, 默认 20\n"
           "  --duration  <秒>       最长运行时长, 默认 1500\n"
           "  --score   <浮点>       置信度阈值, 默认 0.25\n"
           "  --classes 2,3,5,7      COCO 车辆类(默认车/摩托/巴士/卡车)\n"
           "  --save-det             额外保存检测框图\n",
           prog, Params{}.model.c_str(), Params{}.out_dir.c_str());
}

bool parse_roi(const std::string &text, int roi[4]) {
    return sscanf(text.c_str(), "%d,%d,%d,%d", &roi[0], &roi[1], &roi[2], &roi[3]) == 4;
}

// 解析类别列表 "2,3,5,7"
std::set<int> parse_classes(const std::string &text) {
    std::set<int> cls;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t comma = text.find(',', pos);
        const std::string item =
            text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!item.empty()) {
            cls.insert(atoi(item.c_str()));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return cls;
}

std::string timestamp_name() {
    const time_t t = time(nullptr);
    tm tmv{};
    localtime_r(&t, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return buf;
}

// 毫秒序号（同秒多次落图时文件名不冲突）
long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// GStreamer 管线：rtspsrc TCP + mppvideodec 硬解 + appsink NV12（drop=1 永远拿最新帧）
std::string build_pipeline(const std::string &url) {
    return "rtspsrc location=\"" + url +
           "\" protocols=tcp latency=100 ! rtph265depay ! h265parse ! mppvideodec ! "
           "video/x-raw,format=NV12 ! appsink drop=1 max-buffers=1";
}

// NV12 整帧/ROI 转 BGR（x2/y2 传 0 表示到边界；坐标自动偶对齐）
cv::Mat nv12_to_bgr(const cv::Mat &nv12, const int x1, const int y1, const int x2, const int y2) {
    const int h = nv12.rows * 2 / 3;
    const int rx1 = x1 & ~1;
    const int ry1 = y1 & ~1;
    const int rx2 = (x2 > 0 ? std::min(x2, nv12.cols) : nv12.cols) & ~1;
    const int ry2 = (y2 > 0 ? std::min(y2, h) : h) & ~1;
    const int cw = rx2 - rx1;
    const int ch = ry2 - ry1;
    if (cw <= 0 || ch <= 0) {
        return {};
    }
    cv::Mat y_plane = nv12(cv::Rect(rx1, ry1, cw, ch));
    cv::Mat uv_plane = nv12(cv::Rect(rx1, h + ry1 / 2, cw, ch / 2));
    cv::Mat yuv(ch * 3 / 2, cw, CV_8UC1);
    y_plane.copyTo(yuv(cv::Rect(0, 0, cw, ch)));
    uv_plane.copyTo(yuv(cv::Rect(0, ch, cw, ch / 2)));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

}  // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // 行缓冲：nohup 下日志即时可见
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    Params p;
    std::string roi_text;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                printf("错误：--%s 缺少参数\n", what);
                exit(0);
            }
            return argv[++i];
        };
        if (a == "--model") p.model = next("model");
        else if (a == "--url") p.url = next("url");
        else if (a == "--tag") p.tag = next("tag");
        else if (a == "--out") p.out_dir = next("out");
        else if (a == "--roi") roi_text = next("roi");
        else if (a == "--interval") p.interval_ms = atoi(next("interval").c_str());
        else if (a == "--max-saves") p.max_saves = atoi(next("max-saves").c_str());
        else if (a == "--duration") p.duration_s = atoi(next("duration").c_str());
        else if (a == "--score") p.score = static_cast<float>(atof(next("score").c_str()));
        else if (a == "--classes") p.classes = parse_classes(next("classes"));
        else if (a == "--save-det") p.save_det = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
    }
    if (!roi_text.empty()) {
        if (!parse_roi(roi_text, p.roi)) {
            printf("错误：--roi 格式应为 x1,y1,x2,y2\n");
            return 0;
        }
        p.has_roi = true;
    }
    if (p.url.empty()) {
        print_usage(argv[0]);
        return 0;
    }

    // 1. 初始化 RKNN 推理（任务 AUTO；Letterbox 尺寸取模型输入）
    FILE *fm = fopen(p.model.c_str(), "rb");
    if (!fm) {
        printf("[%s] 错误：模型打不开 %s\n", p.tag.c_str(), p.model.c_str());
        return 1;
    }
    fseek(fm, 0, SEEK_END);
    const long fsize = ftell(fm);
    fseek(fm, 0, SEEK_SET);
    std::vector<char> model_data(static_cast<size_t>(fsize));
    const size_t rd = fread(model_data.data(), 1, model_data.size(), fm);
    fclose(fm);
    if (rd != model_data.size()) {
        printf("[%s] 错误：模型读取不完整\n", p.tag.c_str());
        return 1;
    }
    auto config = std::make_shared<yolo::YOLOConfig>(model_data, p.score, p.iou);
    config->device = yolo::InferDevice::NPU;
    // ByteTrack 参数按处理频率放宽：高速车流帧间位移大，IoU 匹配阈值要调低
    config->track_frame_rate = 5;      // 处理频率 5Hz（与 interval 对应）
    config->track_buffer = 60;         // 丢失 60 帧(约12s)内保留轨迹
    config->track_thresh = 0.3f;       // 低分检测进第二轮关联
    config->high_thresh = 0.4f;        // 高分阈值
    config->match_thresh = 0.3f;       // IoU 匹配阈值放宽（防高速运动跟丢）
    yolo::YOLOInference inference(config);
    int code = inference.init();
    if (code != STATUS_SUCCESS) {
        printf("[%s] 错误：模型 init 失败 code=%d\n", p.tag.c_str(), code);
        return 1;
    }
    const auto in_shape = inference.get_input_shape();
    const int box_size = static_cast<int>(in_shape.size() >= 4 ? in_shape[3] : 640);
    yolo::Letterbox letterbox(box_size, 114);

    // 2. 打开 RTSP（GStreamer 硬解）
    const std::string pipeline = build_pipeline(p.url);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("[%s] 错误：RTSP 打开失败\n", p.tag.c_str());
        return 1;
    }
    cv::Mat nv12;
    if (!cap.read(nv12) || nv12.empty()) {
        printf("[%s] 错误：首帧读取失败\n", p.tag.c_str());
        return 1;
    }
    printf("[%s] RTSP 就绪 %dx%d 间隔=%dms 目标=%d张\n", p.tag.c_str(), nv12.cols,
           nv12.rows * 2 / 3, p.interval_ms, p.max_saves);

    // 3. 跟踪摘拍主循环：每个 track 落一张，存够 max_saves 或超时退出
    // 落图时机：优先等车辆完整进画面（框离四边各留 1% 边距），
    //           一直不满足时第 3 次见到该 track 强制落图，保证一车一张不遗漏
    struct TrackState {
        int hits = 0;  // 见到该 track 的次数
        bool saved = false;
    };
    std::map<int, TrackState> tracks;
    int saved = 0;
    long long total_inf = 0;
    auto last_inf = std::chrono::steady_clock::now() - std::chrono::milliseconds(p.interval_ms);
    const auto t_start = std::chrono::steady_clock::now();

    while (g_run && saved < p.max_saves) {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                             t_start)
                .count() >= p.duration_s) {
            printf("[%s] 达到最长运行时长，退出\n", p.tag.c_str());
            break;
        }
        if (!cap.read(nv12) || nv12.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_inf < std::chrono::milliseconds(p.interval_ms)) {
            continue;  // 未到处理间隔，只丢帧保持最新
        }
        last_inf = now;

        cv::Mat bgr = p.has_roi ? nv12_to_bgr(nv12, p.roi[0], p.roi[1], p.roi[2], p.roi[3])
                                : nv12_to_bgr(nv12, 0, 0, 0, 0);
        if (bgr.empty()) {
            continue;
        }
        letterbox.update(bgr.cols, bgr.rows);
        // fill_bgr 契约：先缩放到 dst 尺寸再填充
        cv::Mat scaled;
        cv::resize(bgr, scaled, cv::Size(letterbox.dst_width, letterbox.dst_height));
        letterbox.fill_bgr(scaled.data);

        std::vector<yolo::YOLOResult> results = inference.track(letterbox, &code);
        ++total_inf;
        if (code != STATUS_SUCCESS) {
            printf("[%s] 警告：track 失败 code=%d\n", p.tag.c_str(), code);
            continue;
        }

        // 类别命中且该 track 未落图 → 满足构图条件时落一张原图
        const int margin_x = bgr.cols / 100;
        const int margin_y = bgr.rows / 100;
        for (const auto &r: results) {
            if (r.track_id < 0 || p.classes.count(r.clsId) == 0) {
                continue;
            }
            TrackState &st = tracks[r.track_id];
            ++st.hits;
            if (st.saved) {
                continue;
            }
            const bool inside = r.x > margin_x && r.y > margin_y &&
                                r.x + r.width < bgr.cols - margin_x &&
                                r.y + r.height < bgr.rows - margin_y;
            if (!inside && st.hits < 3) {
                continue;  // 车辆还在画面边缘，等完整进入（最多等到第 3 次）
            }
            st.saved = true;
            ++saved;
            const std::string base = p.out_dir + "/" + p.tag + "_t" + std::to_string(r.track_id) +
                                     "_" + timestamp_name() + "_" +
                                     std::to_string(now_ms() % 100000);
            cv::imwrite(base + ".jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 92});
            if (p.save_det) {
                cv::Mat det = bgr.clone();
                for (const auto &d: results) {
                    cv::rectangle(det, cv::Rect(d.x, d.y, d.width, d.height),
                                  cv::Scalar(0, 255, 0), 2);
                }
                cv::imwrite(base + "_det.jpg", det, {cv::IMWRITE_JPEG_QUALITY, 92});
            }
            printf("[%s] [%d/%d] 新车 track=%d 类别=%d 置信=%.2f 框=%dx%d\n", p.tag.c_str(), saved,
                   p.max_saves, r.track_id, r.clsId, r.score, r.width, r.height);
        }
    }

    printf("[%s] 结束：落图 %d 张，处理 %lld 次推理，见过 track %zu 个\n", p.tag.c_str(), saved,
           total_inf, tracks.size());
    return 0;
}
