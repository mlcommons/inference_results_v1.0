from sys import stdin
import re

mig_profile_regex = re.compile(r"GPU.*Profile ID\ +(\d+).*\{(.+)\}\:\d+")
for line in stdin:
    m = mig_profile_regex.match(line)
    print(",".join([m.group(1)] * len(m.group(2).split(","))))
