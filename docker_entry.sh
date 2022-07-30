#!/bin/sh

# redirect stdout and stderr to files
exec >/results/joblog.log
exec 2>/results/joblog.log

# now run the requested CMD without forking a subprocess
exec "$@"