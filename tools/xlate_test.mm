// Translates every program recorded in a trace directory and compiles the
// result with Metal. Usage: xlate_test <trace-dir> [-v] [-s] [-a] [-d program-id]
//   -v  print the log of failing programs      -d  dump one program's sources
//   -s  translate with soft shadows; also reports how many programs it changed
//   -a  translate with smoothed alpha tests; likewise

#include "../src/shader_xlate.h"

#import <Metal/Metal.h>

#include <fstream>
#include <iostream>
#include <sstream>

static std::string slurp(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

int main(int argc, char **argv)
{
    if (argc < 2) return fprintf(stderr, "usage: %s <trace-dir> [-v] [-d program-id]\n", argv[0]), 2;
    std::string dir = argv[1];
    bool verbose = false;
    XlOptions options;
    int softened = 0, smoothed = 0;
    int dump = -1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) verbose = true;
        if (!strcmp(argv[i], "-s")) options.soft_shadows = true;
        if (!strcmp(argv[i], "-a")) options.smooth_alpha_test = true;
        if (!strcmp(argv[i], "-d") && i + 1 < argc) dump = atoi(argv[++i]);
    }
    const std::map<std::string, int> attribs = {{"inPosition", 0}, {"inNormal", 1},   {"inColor0", 2},
                                                {"inColor1", 3},   {"inTexCoord", 4}, {"inTangent", 5},
                                                {"inBinormal", 6}, {"inBlendWeight", 7}, {"inBlendIndices", 8}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    std::ifstream list(dir + "/programs.txt");
    std::string line, vs, fs;
    int prog = 0, total = 0, failed = 0;
    auto finish = [&] {
        if (vs.empty() || fs.empty()) return;
        total++;
        XlProgram p = xl_translate(slurp(dir + "/" + vs), slurp(dir + "/" + fs), attribs, options);
        softened += p.fs.msl.find("wbm_shadow") != std::string::npos;
        smoothed += p.fs.msl.find("wbm_cover") != std::string::npos;
        std::string error = p.ok ? p.log : "translation failed:\n" + p.log;
        if (p.ok) {
            for (const XlStage *stage : {&p.vs, &p.fs}) {
                NSError *e = nil;
                id<MTLLibrary> lib = [device newLibraryWithSource:@(stage->msl.c_str()) options:nil error:&e];
                if (!lib) error += std::string("Metal: ") + e.localizedDescription.UTF8String + "\n";
            }
        }
        if (!error.empty()) {
            failed++;
            printf("program %d FAILED (%s, %s)\n", prog, vs.c_str(), fs.c_str());
            if (verbose) printf("%s\n", error.c_str());
        }
        if (prog == dump) {
            printf("---- vertex MSL (entry %s, uniform buffer %d, %d bytes)\n%s\n", p.vs.entry.c_str(),
                   p.vs.uniform_buffer_index, p.vs.uniform_buffer_size, p.vs.msl.c_str());
            printf("---- fragment MSL (entry %s, uniform buffer %d, %d bytes)\n%s\n", p.fs.entry.c_str(),
                   p.fs.uniform_buffer_index, p.fs.uniform_buffer_size, p.fs.msl.c_str());
            for (const XlStage *stage : {&p.vs, &p.fs}) {
                for (auto &u : stage->uniforms)
                    printf("uniform %-28s offset %4d %s %dx%d array %d stride %d\n", u.name.c_str(), u.offset,
                           u.is_int ? "int  " : "float", u.columns, u.components, u.array_size, u.array_stride);
                for (auto &s : stage->samplers) printf("sampler %-28s texture %d\n", s.name.c_str(), s.texture_index);
                for (int l : stage->input_locations) printf("input location %d\n", l);
            }
        }
        vs.clear();
        fs.clear();
    };
    while (std::getline(list, line)) {
        char file[256];
        int n;
        if (sscanf(line.c_str(), "program %d", &n) == 1) {
            finish();
            prog = n;
        } else if (sscanf(line.c_str(), " shader %d GL_VERTEX_SHADER %255s", &n, file) == 2) {
            vs = file;
        } else if (sscanf(line.c_str(), " shader %d GL_FRAGMENT_SHADER %255s", &n, file) == 2) {
            fs = file;
        }
    }
    finish();
    printf("%d programs, %d failed\n", total, failed);
    if (options.soft_shadows) printf("%d programs use the soft shadow filter\n", softened);
    if (options.smooth_alpha_test) printf("%d programs have smoothed alpha tests\n", smoothed);
    return failed != 0;
}
