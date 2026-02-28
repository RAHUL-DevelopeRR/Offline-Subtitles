/* Minimal config.h for standalone VLC 3.0 plugin build on MinGW-w64 */
#define HAVE_CONFIG_H 1
#define __PLUGIN__ 1
#define MODULE_STRING "aisub"
#define PACKAGE "vlc"
#define PACKAGE_VERSION "3.0.23"
#define VERSION "3.0.23"
#define PACKAGE_NAME "VLC media player"
#define PACKAGE_STRING "VLC media player 3.0.23"
#define _WIN32_WINNT 0x0601
#define WIN32_LEAN_AND_MEAN 1
#define _FILE_OFFSET_BITS 64

/* Tell vlc_fixups.h that MinGW already provides these */
#define HAVE_MAX_ALIGN_T 1
#define HAVE_POLL 1
#define HAVE_STRUCT_POLLFD 1
#define HAVE_GETPID 1
#define HAVE_ISATTY 1
#define HAVE_NANF 1
#define HAVE_SWAB 1
#define HAVE_SEARCH_H 1
#define HAVE_GETENV 1
#define HAVE_REWIND 1
#define HAVE_STRTOF 1
#define HAVE_ATOF 1
#define HAVE_STRDUP 1
#define HAVE_STRNLEN 1
#define HAVE_STRNDUP 1
#define HAVE_STRTOLL 1
#define HAVE_LLDIV 1
#define HAVE_STRCASECMP 1
#define HAVE_VASPRINTF 1
#define HAVE_ASPRINTF 1
#define HAVE_GMTIME_R 1
#define HAVE_LOCALTIME_R 1
#define HAVE_STRTOK_R 1
#define HAVE_STRCASESTR 1
#define HAVE_STRSEP 1
#define HAVE_INET_PTON 1

/* Forward-declare poll() so vlc_threads.h inline compiles.
   We never actually call vlc_poll() in our plugin. */
struct pollfd;
int poll(struct pollfd *, unsigned, int);

/* VLC platform fixups — must come after HAVE_* defines */
#include <vlc_fixups.h>
