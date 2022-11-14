small = set()
large = set()

small_dict_wrongly_pred = dict()
small_dict_correctly_pred = dict()
large_dict_correctly_pred = dict()
large_dict_wrongly_pred = dict()

threshold_large = list()
threshold_small = list()

with open('log_111022.txt', 'r') as file:
    for line in file:
        split = line.split('\t')

        #
        class_type = split[2]
        predicted_class = split[3][11:]
        correct_class = split[4][9:]
        pos = split[1][3:]
        score = split[-1][7:-1]
        e_val = split[-2][7:]

        if class_type == 'small':
            if correct_class == predicted_class:
                small.add(correct_class)

                if predicted_class in small_dict_correctly_pred:
                    small_dict_correctly_pred[predicted_class].append(
                        [pos, score, e_val, predicted_class, correct_class])
                else:
                    small_dict_correctly_pred[predicted_class] = [[pos, score, e_val, predicted_class, correct_class]]

            elif correct_class != predicted_class:
                if predicted_class in small_dict_wrongly_pred:
                    small_dict_wrongly_pred[predicted_class].append([pos, score, e_val, predicted_class, correct_class])
                else:
                    small_dict_wrongly_pred[predicted_class] = [[pos, score, e_val, predicted_class, correct_class]]

        elif class_type == 'large':
            if correct_class == predicted_class:
                large.add(correct_class)

                if predicted_class in large_dict_correctly_pred:
                    large_dict_correctly_pred[predicted_class].append(
                        [pos, score, e_val, predicted_class, correct_class])
                else:
                    large_dict_correctly_pred[predicted_class] = [[pos, score, e_val, predicted_class, correct_class]]

            elif correct_class != predicted_class:
                if predicted_class in large_dict_wrongly_pred:
                    large_dict_wrongly_pred[predicted_class].append([pos, score, e_val, predicted_class, correct_class])
                else:
                    large_dict_wrongly_pred[predicted_class] = [[pos, score, e_val, predicted_class, correct_class]]

for item in large:
    correcrly_pred = list()
    wrongly_pred = list()
    if item in large_dict_correctly_pred:
        for item1 in large_dict_correctly_pred[item]:
            correcrly_pred.append(item1[1])

    if item in large_dict_wrongly_pred:
        for item2 in large_dict_wrongly_pred[item]:
            wrongly_pred.append(item2[1])

    if len(wrongly_pred) != 0:
        highest_negative_score = max(wrongly_pred)
    else:
        highest_negative_score = 0

    correcrly_pred.sort()

    for pred_score in correcrly_pred:
        if float(pred_score) > float(highest_negative_score):
            temp_threshold = pred_score
            threshold_large.append((item, temp_threshold))
            break

for item in small:
    correcrly_pred = list()
    wrongly_pred = list()
    if item in small_dict_correctly_pred:
        for item1 in small_dict_correctly_pred[item]:
            correcrly_pred.append(item1[1])

    if item in small_dict_wrongly_pred:
        for item2 in small_dict_wrongly_pred[item]:
            wrongly_pred.append(item2[1])

    if len(wrongly_pred) != 0:
        highest_negative_score = max(wrongly_pred)
    else:
        highest_negative_score = 0

    correcrly_pred.sort()

    for pred_score in correcrly_pred:
        if float(pred_score) > float(highest_negative_score):
            temp_threshold = pred_score
            threshold_small.append((item, temp_threshold))
            break

# Writing threshold as files
with open ('thresholds_large', 'a') as file:
    for item in threshold_large:
        print(item[0], '\t', item[1], file=file, sep='')

with open ('thresholds_small', 'a') as file:
    for item in threshold_small:
        print(item[0], '\t', item[1], file=file, sep='')