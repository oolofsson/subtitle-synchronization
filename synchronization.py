from evaluation import *

def is_unsynchronized(predicted, subtitles_presence_array, start, end, steps):
    current_similarity = get_similarity(predicted, subtitles_array_sec[start:end])
    # search forwards
    i = 1
    while i < steps:
        sim = get_similarity(predicted, subtitles_presence_array[start + i:end + i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1

    # search backwards
    i = 1
    while i < steps:
        sim = get_similarity(predicted, subtitles_presence_array[start - i:end - i])
        #print("i is: " + str(i) + " and sim is: " + str(sim))
        if current_similarity < sim:
            return True
        i += 1
    return False

def synchronize(predicted, subtitles_presence_array, start, end, steps):
    # generate new srt file
    pass
