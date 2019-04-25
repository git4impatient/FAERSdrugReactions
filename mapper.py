#!/usr/bin/env python
# copyright (c) 2019 Marty Lurie   Sample code not supported
# for use with hadoop streaming

import sys
import md5

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    row = line.split("$")
    print '%s.%s\t%s' % (row[0],row[3],row[1])
