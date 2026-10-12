#include "trace.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

const char *const g_fn_names[FN_COUNT] = {
#define GL_DIRECT(n) "gl" #n,
#define GL_GLEW(n) "gl" #n,
#include "gl_api.def"
    "glewInit",
    "SDL_CreateWindow",
    "SDL_GL_CreateContext",
    "SDL_GL_DeleteContext",
    "SDL_GL_MakeCurrent",
    "SDL_GL_SetAttribute",
    "SDL_GL_SetSwapInterval",
    "SDL_GL_SwapWindow",
};

std::atomic<bool> g_verbose{false};
std::atomic<uint32_t> g_frame{0};
std::atomic<uint64_t> g_frame_draws{0};
bool g_glerr = false;
bool g_native = false;
bool g_metal = false;
bool g_fast = false;
const char *g_out_dir = "";

namespace {

enum { HIST_CAP_PER_FN = 20000, SUMMARY_TOP = 200, SUMMARY_EVERY = 300 };

std::string s_out;
FILE *s_log;
std::mutex s_log_mu;
uint64_t s_log_bytes, s_log_cap = 256ull << 20, s_log_dropped;

std::atomic<uint64_t> s_calls[FN_COUNT];
std::atomic<uint32_t> s_thread_mask[FN_COUNT];
std::atomic<uint64_t> s_frame_calls{0};
std::atomic<uint64_t> s_foreign[FN_COUNT];

struct ThreadInfo {
    bool is_main;
    char name[64];
};
std::mutex s_thread_mu;
std::vector<ThreadInfo> s_threads;
int s_context_thread = -1;
std::string s_context_info;

struct HKey {
    uint64_t k[4];
    int fn;
    bool operator==(const HKey &o) const { return fn == o.fn && !memcmp(k, o.k, sizeof k); }
};
struct HKeyHash {
    size_t operator()(const HKey &a) const
    {
        uint64_t h = (uint64_t)a.fn * 0x9E3779B97F4A7C15ull;
        for (uint64_t v : a.k) h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};
struct HVal {
    uint64_t count = 0;
    std::string text;
};
std::mutex s_hist_mu;
std::unordered_map<HKey, HVal, HKeyHash> s_hist;
uint32_t s_hist_n[FN_COUNT];
uint64_t s_hist_over[FN_COUNT];

std::vector<std::pair<uint32_t, uint32_t>> s_verbose_ranges; // WBM_TRACE_FRAMES
volatile sig_atomic_t s_snap_requested;

std::mutex s_frame_mu;
std::vector<float> s_frame_ms;
std::chrono::steady_clock::time_point s_last_swap;
uint64_t s_total_calls, s_max_calls, s_total_draws, s_max_draws;

bool in_verbose_ranges(uint32_t f)
{
    for (auto &r : s_verbose_ranges)
        if (f >= r.first && f <= r.second) return true;
    return false;
}

void parse_ranges(const char *s)
{
    while (s && *s) {
        char *end;
        unsigned long a = strtoul(s, &end, 10), b = a;
        if (end == s) break;
        if (*end == '-') {
            s = end + 1;
            b = strtoul(s, &end, 10);
        }
        s_verbose_ranges.emplace_back((uint32_t)a, (uint32_t)b);
        s = (*end == ',') ? end + 1 : end;
        if (*end && *end != ',') break;
    }
}

void mkdirs(const std::string &path)
{
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') mkdir(path.substr(0, i).c_str(), 0755);
}

void write_line(const char *line, bool force)
{
    std::lock_guard<std::mutex> lock(s_log_mu);
    if (!s_log) return;
    if (!force && s_log_bytes > s_log_cap) {
        if (!s_log_dropped++)
            fputs("!! log size cap reached (WBM_LOG_MAX_MB); only verbose frames are logged from here on\n", s_log);
        return;
    }
    s_log_bytes += fprintf(s_log, "%s\n", line);
}

// fps.log: one line per second of wall time. Called with s_frame_mu held.
void fps_tick(std::chrono::steady_clock::time_point now, float frame_ms)
{
    static FILE *f = fopen((s_out + "/fps.log").c_str(), "w");
    static auto start = now, window = now;
    static int frames;
    static float worst;
    if (!f) return;
    frames++;
    worst = std::max(worst, frame_ms);
    float span = std::chrono::duration<float>(now - window).count();
    if (span < 1.0f) return;
    fprintf(f, "t=%6.0fs  frame %7u  %6.1f fps  avg %6.2f ms  worst %7.2f ms\n",
            std::chrono::duration<float>(now - start).count(), g_frame.load(), frames / span, 1000.0f * span / frames, worst);
    fflush(f);
    window = now;
    frames = 0;
    worst = 0;
}

void on_sigusr1(int) { s_snap_requested = 1; }

std::string thread_list(uint32_t mask)
{
    std::string s;
    for (int i = 0; i < 32; i++)
        if (mask & (1u << i)) s += (s.empty() ? "t" : ",t") + std::to_string(i);
    return s;
}

} // namespace

int wb_thread()
{
    static thread_local int idx = -1;
    if (idx < 0) {
        ThreadInfo info{};
        info.is_main = pthread_main_np() != 0;
        pthread_getname_np(pthread_self(), info.name, sizeof info.name);
        std::lock_guard<std::mutex> lock(s_thread_mu);
        idx = (int)s_threads.size();
        s_threads.push_back(info);
    }
    return idx;
}

void wb_count(int fn)
{
    s_calls[fn].fetch_add(1, std::memory_order_relaxed);
    s_frame_calls.fetch_add(1, std::memory_order_relaxed);
    uint32_t bit = 1u << std::min(wb_thread(), 31);
    if (!(s_thread_mask[fn].load(std::memory_order_relaxed) & bit)) s_thread_mask[fn].fetch_or(bit);
}

bool wb_count_foreign(int fn)
{
    return s_foreign[fn].fetch_add(1, std::memory_order_relaxed) == 0;
}

void wb_log(int fn, const char *fmt, ...)
{
    char buf[4096];
    int n = snprintf(buf, sizeof buf, "f%06u t%d %s ", g_frame.load(), wb_thread(), g_fn_names[fn]);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf + n, sizeof buf - n, fmt, ap);
    va_end(ap);
    write_line(buf, g_verbose.load(std::memory_order_relaxed));
}

void wb_note(const char *fmt, ...)
{
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    write_line(buf, true);
}

void wb_flush()
{
    std::lock_guard<std::mutex> lock(s_log_mu);
    if (s_log) fflush(s_log);
}

bool wb_hist(int fn, uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3)
{
    HKey key{{k0, k1, k2, k3}, fn};
    std::lock_guard<std::mutex> lock(s_hist_mu);
    auto it = s_hist.find(key);
    if (it != s_hist.end()) {
        it->second.count++;
        return false;
    }
    if (s_hist_n[fn] >= HIST_CAP_PER_FN) {
        s_hist_over[fn]++;
        return false;
    }
    s_hist_n[fn]++;
    s_hist[key].count = 1;
    return true;
}

void wb_hist_text(int fn, uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    HKey key{{k0, k1, k2, k3}, fn};
    std::lock_guard<std::mutex> lock(s_hist_mu);
    s_hist[key].text = buf;
}

void wb_set_context_thread()
{
    int t = wb_thread();
    std::lock_guard<std::mutex> lock(s_thread_mu);
    s_context_thread = t;
}

void wb_set_context_info(const char *text)
{
    std::lock_guard<std::mutex> lock(s_thread_mu);
    s_context_info = text;
}

void wb_frame_end()
{
    auto now = std::chrono::steady_clock::now();
    uint64_t calls = s_frame_calls.exchange(0), draws = g_frame_draws.exchange(0);
    bool was_verbose = g_verbose.load();
    uint32_t ended = g_frame.load();
    {
        std::lock_guard<std::mutex> lock(s_frame_mu);
        if (ended > 0) {
            if (s_frame_ms.size() < 2000000)
                s_frame_ms.push_back(std::chrono::duration<float, std::milli>(now - s_last_swap).count());
            s_total_calls += calls;
            s_total_draws += draws;
            s_max_calls = std::max(s_max_calls, calls);
            s_max_draws = std::max(s_max_draws, draws);
        }
        if (ended > 0) fps_tick(now, std::chrono::duration<float, std::milli>(now - s_last_swap).count());
        s_last_swap = now;
    }
    if (was_verbose) wb_note("=== frame %u end: %llu calls, %llu draws ===", ended, (unsigned long long)calls, (unsigned long long)draws);

    uint32_t next = ended + 1;
    bool verbose = in_verbose_ranges(next);
    if (s_snap_requested) {
        s_snap_requested = 0;
        verbose = true;
    }
    g_frame.store(next);
    g_verbose.store(verbose);
    if (verbose) wb_note("=== frame %u begin (verbose) ===", next);
    wb_flush();
    if (was_verbose || next % SUMMARY_EVERY == 0) wb_write_summary();
}

void wb_write_summary()
{
    static std::mutex mu;
    std::lock_guard<std::mutex> only_one(mu);
    std::string tmp = s_out + "/summary.txt.tmp", path = s_out + "/summary.txt";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return;

    fprintf(f, "warband-metal trace summary\nframes presented: %u\n\n", g_frame.load());
    {
        std::lock_guard<std::mutex> lock(s_thread_mu);
        fprintf(f, "== GL context ==\n%s\n", s_context_info.empty() ? "(not created yet)\n" : s_context_info.c_str());
        fprintf(f, "== threads that made traced calls ==\n");
        for (size_t i = 0; i < s_threads.size(); i++)
            fprintf(f, "t%zu%s%s%s%s\n", i, s_threads[i].is_main ? " main" : "",
                    (int)i == s_context_thread ? " [created the GL context]" : "", s_threads[i].name[0] ? " name=" : "",
                    s_threads[i].name);
    }
    {
        std::lock_guard<std::mutex> lock(s_frame_mu);
        fprintf(f, "\n== frames ==\n");
        if (!s_frame_ms.empty()) {
            std::vector<float> v = s_frame_ms;
            std::sort(v.begin(), v.end());
            auto pct = [&](double p) { return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
            size_t n = v.size();
            double total = 0;
            for (float ms : v) total += ms;
            fprintf(f, "mode: %s\naverage: %.1f fps (%.2f ms)\n", g_native ? "native (no tracing)" : g_fast ? "metal (no tracing)" : g_metal ? "metal, traced" : "trace", 1000.0 * v.size() / total,
                    total / v.size());
            fprintf(f, "frame time ms: p50 %.2f  p90 %.2f  p99 %.2f  max %.2f  (%zu frames)\n", pct(0.50), pct(0.90),
                    pct(0.99), v.back(), n);
            fprintf(f, "traced calls per frame: avg %.0f  max %llu\n", (double)s_total_calls / n,
                    (unsigned long long)s_max_calls);
            fprintf(f, "draw calls per frame:   avg %.0f  max %llu\n", (double)s_total_draws / n,
                    (unsigned long long)s_max_draws);
        } else {
            fprintf(f, "(none yet)\n");
        }
    }

    fprintf(f, "\n== calls per function ==\n");
    std::string never;
    for (int i = 0; i < FN_COUNT; i++) {
        uint64_t c = s_calls[i].load();
        if (!c) {
            never += std::string("  ") + g_fn_names[i] + "\n";
            continue;
        }
        fprintf(f, "%-28s %12llu  %s\n", g_fn_names[i], (unsigned long long)c, thread_list(s_thread_mask[i].load()).c_str());
    }
    fprintf(f, "\n== never called ==\n%s", never.empty() ? "  (none)\n" : never.c_str());

    fprintf(f, "\n== calls from other libraries (forwarded, not traced) ==\n");
    for (int i = 0; i < FN_COUNT; i++)
        if (uint64_t c = s_foreign[i].load()) fprintf(f, "%-28s %12llu\n", g_fn_names[i], (unsigned long long)c);

    struct Row {
        uint64_t count;
        std::string text;
    };
    std::vector<std::vector<Row>> rows(FN_COUNT);
    uint64_t over[FN_COUNT];
    {
        std::lock_guard<std::mutex> lock(s_hist_mu);
        for (auto &e : s_hist) rows[e.first.fn].push_back({e.second.count, e.second.text});
        memcpy(over, s_hist_over, sizeof over);
    }
    fprintf(f, "\n== distinct argument tuples (count, arguments) ==\n");
    for (int i = 0; i < FN_COUNT; i++) {
        if (rows[i].empty()) continue;
        std::sort(rows[i].begin(), rows[i].end(), [](const Row &a, const Row &b) {
            return a.count != b.count ? a.count > b.count : a.text < b.text;
        });
        fprintf(f, "\n%s: %zu distinct\n", g_fn_names[i], rows[i].size());
        for (size_t j = 0; j < rows[i].size() && j < SUMMARY_TOP; j++)
            fprintf(f, "  %10llu  %s\n", (unsigned long long)rows[i][j].count, rows[i][j].text.c_str());
        if (rows[i].size() > SUMMARY_TOP) fprintf(f, "  ... %zu more\n", rows[i].size() - SUMMARY_TOP);
        if (over[i]) fprintf(f, "  %llu calls not classified (tuple table full)\n", (unsigned long long)over[i]);
    }

    wb_gl_dump_state(f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

static void wb_init()
{
    // Children of the game (crash reporters, Steam helpers) must not inherit the shim.
    unsetenv("DYLD_INSERT_LIBRARIES");

    // The default is what a player wants: Metal, no tracing. "trace" (OpenGL
    // with tracing) and "native" (OpenGL untouched, frame times only) are for
    // development and comparison.
    const char *mode = getenv("WBM_MODE");
    if (mode && !strcmp(mode, "trace")) {
    } else if (mode && !strcmp(mode, "native")) {
        g_native = true;
    } else {
        if (mode && *mode && strcmp(mode, "metal")) fprintf(stderr, "wbmetal: unknown WBM_MODE '%s', using metal\n", mode);
        g_metal = true;
        const char *trace = getenv("WBM_TRACE");
        g_fast = !(trace && atoi(trace) != 0);
    }

    // Without WBM_OUT the log of the latest run goes where macOS keeps logs.
    const char *out = getenv("WBM_OUT");
    const char *home = getenv("HOME");
    s_out = out && *out ? out : std::string(home ? home : "/tmp") + "/Library/Logs/warband-metal";
    mkdirs(s_out);
    char abs[PATH_MAX];
    if (realpath(s_out.c_str(), abs)) s_out = abs; // the game changes its working directory
    if (!g_fast && !g_native) mkdirs(s_out + "/shaders");
    mkdirs(s_out + "/frames");
    g_out_dir = s_out.c_str();

    s_log = fopen((s_out + "/trace.log").c_str(), "w");
    if (!s_log) {
        fprintf(stderr, "wbmetal: cannot write to %s\n", s_out.c_str());
        return;
    }
    setvbuf(s_log, nullptr, _IOFBF, 1 << 20);

    parse_ranges(getenv("WBM_TRACE_FRAMES"));
    if (const char *v = getenv("WBM_LOG_MAX_MB")) s_log_cap = strtoull(v, nullptr, 10) << 20;
    if (const char *v = getenv("WBM_GLERR")) g_glerr = atoi(v) != 0 && !g_metal; // no OpenGL context to ask in Metal mode
    g_verbose.store(in_verbose_ranges(0));

    struct sigaction sa {};
    sa.sa_handler = on_sigusr1;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa, nullptr);

    atexit([] {
        wb_note("=== process exit ===");
        wb_write_summary();
        wb_flush();
    });

    wb_note("wbmetal %s, pid %d", g_native ? "native mode (frame timing only, GL calls untouched)" : g_fast ? "Metal backend, untraced" : g_metal ? "Metal backend, traced" : "trace shim", getpid());
    wb_note("line format: f<frame> t<thread> <function> <arguments>");
    wb_flush();
    fprintf(stderr, "wbmetal: tracing to %s\n", s_out.c_str());
}

// A dynamic initializer rather than __attribute__((constructor)): initializers
// in one file run in order, so this runs after the globals above are built.
// Keep it last in the file.
static const int s_initialized = (wb_init(), 0);
