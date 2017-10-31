import numpy as np

def cross_val(model, data, labels, k ):
    calculations = {}
    examples_size = len(data)
    fold_size = examples_size / k
    calculations['scores'] = []
    calculations['min_score'] = 1
    calculations['max_score'] = 0
    calculations['mean'] = 0

    start = 0

    for fold_num in range(k):
        end = int(start + fold_size)

        x_test = data[start:end]
        y_test = labels[start:end]


        data_start_split = data[0:start]
        data_end_split = data[end:examples_size]


        if data_start_split.size == 0:
            x_train = data_end_split
        elif data_end_split.size == 0:
            x_train = data_start_split
        else:
            x_train = np.concatenate((data_start_split,data_end_split))

        label_start_split = labels[0:start]
        label_end_split = labels[end:examples_size]

        if label_start_split.size == 0:
            y_train = label_end_split
        elif label_end_split.size == 0:
            y_train = label_start_split
        else:
            y_train = np.concatenate((label_start_split, label_end_split))

        start += int(fold_size)

        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        calculations['min_score'] = min(calculations['min_score'], score)
        calculations['max_score'] = max(calculations['max_score'], score)
        calculations['mean'] += score
        calculations['scores'].append(score)


    calculations['mean'] /= k

    sum_scores = 0

    for score in calculations['scores']:
        sum_scores += pow(abs(score - calculations['mean']),2)
        calculations['sd'] = np.sqrt(sum_scores / k)


    sorted_scores = np.sort(np.copy(calculations['scores']))
    calculations['median'] = find_median_in_sorted_arr(sorted_scores)
    calculations['lower_quartile'] = find_median_in_sorted_arr(sorted_scores[0:int(len(sorted_scores)/2)])
    calculations['higher_quartile'] = find_median_in_sorted_arr(sorted_scores[int(len(sorted_scores)/2 + 1):int(len(sorted_scores))])

    return calculations


def find_median_in_sorted_arr(arr):
    length = len(arr)
    if len(arr) % 2 == 0:
        return (arr[int(length/2)] + arr[int(length/2 - 1)]) / 2
    else:
        return arr[int(length / 2)]