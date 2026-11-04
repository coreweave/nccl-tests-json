/*************************************************************************
 * Build-time extension hooks.
 *
 * These macros let an extension linked in at build time observe a run. At
 * startup the extension receives the command line and the environment,
 * filtered the same way as in the JSON report. During the run it receives
 * the run configuration, the result for each message size, every device,
 * the closing summary, and any errors. On rank 0 it picks a run ID for every
 * process to share, and at the end it collects text from every process.
 * Setting EXT_DIR in the Makefile defines NCCL_TESTS_EXTENSION, which
 * includes <nccl_tests_ext.h> from EXT_DIR/include and enables the hooks.
 * When EXT_DIR is unset, the hooks compile to nothing and the tests behave
 * exactly as they would without them.
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
