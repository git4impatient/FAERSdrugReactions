#!/usr/bin/env python
# this should be used as the reducer stage in a map-reduce-streaming job
# or run it from the command line on smaller data

import sys
import md5
import hashlib

oldisr=""
medscount=0
medslist=""
# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    isrdrug = line.split("\t")
    isr_drug= isrdrug[0].split(".")
    isr= isr_drug[0]
    if (isr != oldisr):
        # generate hash and normalize it for the drug list
        # isr | medcount | medhash | medslist
        drugslisthash=hashlib.md5(medslist)
        medhash= float((int(drugslisthash.hexdigest(),16)%10**9))/10**9
        print '%s|%i|%f|%s' % (oldisr, medscount, medhash, medslist )
        oldisr=isr
        medscount=0
        medslist=""
        medscount = medscount+1
        medslist= medslist +"*"+ isr_drug[1]
    else :
        medscount = medscount+1
        medslist= medslist +"*"+ isr_drug[1]

