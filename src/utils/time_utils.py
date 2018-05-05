

def time_diff(start, end):
    _sec = end - start
    _min = int(_sec / 60.)
    if _min > 0:
        return "%d mins" % _min
    else:
        return "%d secs" % _sec