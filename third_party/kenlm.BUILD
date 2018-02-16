licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "kenlm",
    hdrs = glob(['lm/*.hh', 'util/*.hh', 'util/double-conversion/*.h'],
                exclude = []),
    srcs = glob(['lm/*.cc', 'util/*.cc', 'util/double-conversion/*.cc'],
                exclude = ['*/*test.cc', '*/*main.cc']),
    includes = ['include'],
    defines = ['KENLM_MAX_ORDER=6'],
    copts = ['-std=c++03', '-fexceptions'],
    visibility = ["//visibility:public"],
)
