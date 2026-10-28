/*************************************************************************
 * Build-time extension hooks.
 *
 * Public builds define nothing here. Every NCCL_TESTS_EXT(...) compiles to an
 * empty statement and NCCL_TESTS_EXT_ENABLED is 0, so control flow is unchanged.
 * Builds that set EXT_DIR in the Makefile define NCCL_TESTS_EXTENSION and provide
 * <nccl_tests_ext.h> with the hook functions. The hooks receive the values that the
 * test already prints. They never alter output or exit codes.
 ************************************************************************/
#ifndef __EXT_HOOKS_H__
#define __EXT_HOOKS_H__

#ifdef NCCL_TESTS_EXTENSION
#include <nccl_tests_ext.h>
#define NCCL_TESTS_EXT_ENABLED 1
// Only the main thread of the reporting process reports. run() sets is_main_thread.
#define NCCL_TESTS_EXT(call)        do { if (is_main_thread) { call; } } while (0)
// For hooks that run on every process, such as init, the run id exchange and finalize.
#define NCCL_TESTS_EXT_ALWAYS(call) do { call; } while (0)
#else
#define NCCL_TESTS_EXT_ENABLED 0
#define NCCL_TESTS_EXT(call)        do { } while (0)
#define NCCL_TESTS_EXT_ALWAYS(call) do { } while (0)
#endif

#endif
