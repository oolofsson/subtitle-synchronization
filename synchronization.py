from evaluation import *
from preprocessing import *

def is_unsynchronized(predicted, subtitles_presence_array, start, end, steps):
    current_similarity = get_similarity(predicted, subtitles_presence_array[start:end])
    # search forwards
    i = 1
    while i < steps:
        if current_similarity < get_similarity(predicted, subtitles_presence_array[start + i:end + i]) or current_similarity < get_similarity(predicted, subtitles_presence_array[start - i:end - i]):
            return True
        i += 1
    return False

def synchronize(predicted, subtitles_presence_array, start, end, steps):
    best_shift = find_best_shift(predicted, subtitles_presence_array, start, end, steps)
    print("start sync")
    if best_shift != 0:
        print("syncing at stepsize: ", steps)
        shift(subtitles_presence_array, start, end, best_shift)
        synchronize(predicted, subtitles_presence_array, start, int((end - start) / 2) + steps, steps)
        synchronize(predicted, subtitles_presence_array, int((end - start) / 2) + steps, end, steps)


def shift(subtitles_presence_array, start, end, best_shift):
    # and perform srt shifting
    with open("datasets/NGPlus_163151_183450_no_shift2.srt") as file:
        srt_string = file.read()
        subtitles = srt.parse(srt_string)
        for subtitle in subtitles:
            print(subtitle)

        print(str(subtitles))


    for i in range(start, end):
        subtitles_presence_array[i] = subtitles_presence_array[i + best_shift]
    print("best shift is", best_shift)

def find_best_shift(predicted, subtitles_presence_array, start, end, steps):
    # generate new srt file
    highest_similarity = get_similarity(predicted[start-steps:end-steps], subtitles_presence_array[start:end])
    # search shift
    i = 1
    best_shift = 0
    while i < steps:
        sim_forwards = get_similarity(predicted[start-steps:end-steps], subtitles_presence_array[start + i:end + i])
        sim_backwards = get_similarity(predicted[start-steps:end-steps], subtitles_presence_array[start - i:end - i])
        if highest_similarity < sim_forwards:
            highest_similarity = sim_forwards
            best_shift = i
        if highest_similarity < sim_backwards:
            highest_similarity = sim_backwards
            best_shift = -i
        i += 1
    return best_shift
