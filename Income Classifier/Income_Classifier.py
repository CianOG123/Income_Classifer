'''

Author Cian O'Gorman

Listing of attributes:

0. Age: Number.
1. Workclass (Discrete): Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
Without-pay, Never-worked.
4. Education-number: Number -- indicates level of education.
5. Marital-status: Can be one of -- Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
Married-spouse-absent, Married-AF-spouse.
6. Occupation (Discrete): Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
Armed-Forces.
7. Relationship (Discrete): Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
8. Race (Discrete): White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
9. Sex: Either Female or Male.
10. Capital-gain: Number.
11. Capital-loss: Number.
12. Hours-per-week: Number.
14. Outcome for this record: Can be >50K or <=50K.
'''

import requests

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

TEST_TRIAL_SPLIT = .75
DISCRETE_TOLERANCE = .05

WORK_CLASS_TYPES = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
MARITAL_STATUS_TYPES = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
OCCUPATION_TYPES = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                        "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", 
                        "Armed-Forces"]
RELATIONSHIP_TYPES = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
RACE_TYPES = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
SEX_TYPES = ["Female", "Male"]

def get_url_data(url):
    try:
        data = requests.get(url)
        if 200 <= data.status_code <= 299:
            if data.headers["Content-Type"] and data.headers["Content-Type"] in ("application/x-httpd-php"):
                return data.text
            else:
                raise ValueError(f"No content'")
        else:
            raise ValueError(f"Bad status code")
    except Exception as e:
        print("\nError encountered: ", e)

def get_clean_data(url):
    bad_records = 0
    cleaned_dataset = []
    data = get_url_data(url)
    data = data.split("\n")
    for record in data:
        try:
            record = record.strip().split(", ")
            if not record:
                raise ValueError("Empty Record")
            if(len(record) >= 11):
                del record[2:4]
                del record[11]
            else:
                raise ValueError("Missing Values");
            for i in range(len(record)):
                if(record[i].isnumeric()):
                    record[i] = int(record[i])
            cleaned_dataset.append(tuple(record))
        except ValueError as val_err:
            bad_records += 1
            print(f"Record {record[0]} rejected: {val_err}")
            continue
    return tuple(cleaned_dataset)

def create_classifier(training_dataset):
    below_50_count = 0
    above_50_count = 0
    classifier_mid_points = [0] * 11
    # Age
    total_above = [0] * 11
    total_below = [0] * 11
    # Work Class
    total_above[1] = [0] * 8
    total_below[1] = [0] * 8
    # Marital Status
    total_above[3] = [0] * 7
    total_below[3] = [0] * 7
    # Occupation
    total_above[4] = [0] * 14
    total_below[4] = [0] * 14
    # Relationship
    total_above[5] = [0] * 6
    total_below[5] = [0] * 6
    # Race
    total_above[6] = [0] * 5
    total_below[6] = [0] * 5
    # Sex
    total_above[7] = [0] * 2
    total_below[7] = [0] * 2

    for record in training_dataset:
        is_above = False
        if(record[11] == ">50K"):
            is_above = True
            above_50_count += 1
            # Age
            total_above[0] += sum_continuous(record[0])
            # Education Status
            total_above[2] += sum_continuous(record[2])
            # Capital Gain
            total_above[8] += sum_continuous(record[8])
            # Capital Loss
            total_above[9] += sum_continuous(record[9])
            # Hours per week
            total_above[10] += sum_continuous(record[10])
        else:
            is_above = False
            below_50_count += 1
            # Age
            total_below[0] += sum_continuous(record[0])
            # Education Status
            total_below[2] += sum_continuous(record[2])
            # Capital Gain
            total_below[8] += sum_continuous(record[8])
            # Capital Loss
            total_below[9] += sum_continuous(record[9])
            # Hours per week
            total_below[10] += sum_continuous(record[10])
        total_above[1], total_below[1] = calculate_above_below(record[1], WORK_CLASS_TYPES, is_above, total_above[1], total_below[1])        
        total_above[3], total_below[3] = calculate_above_below(record[3], MARITAL_STATUS_TYPES, is_above, total_above[3], total_below[3])           
        total_above[4], total_below[4] = calculate_above_below(record[4], OCCUPATION_TYPES, is_above, total_above[4], total_below[4])   
        total_above[5], total_below[5] = calculate_above_below(record[5], RELATIONSHIP_TYPES, is_above, total_above[5], total_below[5])   
        total_above[6], total_below[6] = calculate_above_below(record[6], RACE_TYPES, is_above, total_above[6], total_below[6])    
        total_above[7], total_below[7] = calculate_above_below(record[7], SEX_TYPES, is_above, total_above[7], total_below[7])
    classifier_mid_points = calculate_averages(classifier_mid_points, total_above, total_below, above_50_count, below_50_count)
    return tuple(classifier_mid_points)

def calculate_averages(classifier_mid_points, total_above, total_below, above_50_count, below_50_count):
    # Continuous
    # Age
    classifier_mid_points[0] = find_continuous_average(total_above[0], total_below[0], above_50_count, below_50_count)
    # Education Status
    classifier_mid_points[2] = find_continuous_average(total_above[2], total_below[2], above_50_count, below_50_count)
    # Capital Gain
    classifier_mid_points[8] = find_continuous_average(total_above[8], total_below[8], above_50_count, below_50_count)
    # Capital Loss
    classifier_mid_points[9] = find_continuous_average(total_above[9], total_below[9], above_50_count, below_50_count)
    # Hours Per Week
    classifier_mid_points[10] = find_continuous_average(total_above[10], total_below[10], above_50_count, below_50_count)
    # Discrete
    classifier_mid_points[1] = find_discrete_average(WORK_CLASS_TYPES, total_above[1], total_below[1], above_50_count, below_50_count)
    classifier_mid_points[3] = find_discrete_average(MARITAL_STATUS_TYPES, total_above[3], total_below[3], above_50_count, below_50_count)
    classifier_mid_points[4] = find_discrete_average(OCCUPATION_TYPES, total_above[4], total_below[4], above_50_count, below_50_count)
    classifier_mid_points[5] = find_discrete_average(RELATIONSHIP_TYPES, total_above[5], total_below[5], above_50_count, below_50_count)
    classifier_mid_points[6] = find_discrete_average(RACE_TYPES, total_above[6], total_below[6], above_50_count, below_50_count)
    classifier_mid_points[7] = find_discrete_average(SEX_TYPES, total_above[7], total_below[7], above_50_count, below_50_count)
    return classifier_mid_points

def calculate_above_below(record, data_type, is_above, total_above, total_below):
    if(record != '?'):
        index = data_type.index(record, 0, len(data_type))
        if(index >= 0 and index <= len(data_type)):
            if(is_above == True):
                total_above[index] = 1
            else:
                total_below[index] = 1
    return total_above, total_below

def sum_continuous(record):
    if(record != '?'):
        return record

def find_discrete_average(data_type, total_above, total_below, above_50_count, below_50_count):
    classifier_mid_points = [0] * len(data_type)
    for i in range(len(data_type)):
        classifier_mid_points[i] = find_continuous_average(total_above[i], total_below[i], above_50_count, below_50_count)
    return classifier_mid_points

def find_continuous_average(total_above, total_below, above_50_count, below_50_count):
    above_50 = total_above / above_50_count
    below_50 = total_below / below_50_count
    return (above_50 + below_50) / 2

def continuous_compare(record, classifier_mid_points):
    if(record >= classifier_mid_points):
        return True

def discrete_compare(record, COMPARE_TYPES, classifier_mid_points):
    if(record != '?'):
        index = COMPARE_TYPES.index(record, 0, len(COMPARE_TYPES))
        if(index >= 0 and index <= len(COMPARE_TYPES)):
            if(classifier_mid_points[index] > DISCRETE_TOLERANCE):
                return True

def test_classifier(testing_dataset, classifier_mid_points):
    predicted_correct = 0
    total_tests = 0
    for record in testing_dataset:
        is_above = [False] * len(record)
        # Age
        is_above[0] = continuous_compare(record[0], classifier_mid_points[0])
        # Work Class
        is_above[1] = discrete_compare(record[1], WORK_CLASS_TYPES, classifier_mid_points[1])
        # Education Status
        is_above[2] = continuous_compare(record[2], classifier_mid_points[2])
        # Marital Status
        is_above[3] = discrete_compare(record[3], MARITAL_STATUS_TYPES, classifier_mid_points[3])
        # Occupation
        is_above[4] = discrete_compare(record[4], OCCUPATION_TYPES, classifier_mid_points[4])
        # Relationship
        is_above[5] = discrete_compare(record[5], RELATIONSHIP_TYPES, classifier_mid_points[5])
        # Race
        is_above[6] = discrete_compare(record[6], RACE_TYPES, classifier_mid_points[6])
        # Sex
        is_above[7] = discrete_compare(record[7], SEX_TYPES, classifier_mid_points[7])
        # Capital Gain
        is_above[8] = continuous_compare(record[8], classifier_mid_points[8])
        # Capital Loss
        is_above[9] = continuous_compare(record[9], classifier_mid_points[9])
        # hours Per Week
        is_above[10] = continuous_compare(record[10], classifier_mid_points[10])
        total_tests, predicted_correct = verify(is_above, record, predicted_correct, total_tests)
    return total_tests, predicted_correct

def verify(is_above, record, predicted_correct, total_tests):
    true_count = 0
    for j in range(len(is_above)):
        if(is_above == True):
            true_count += 1
    if(true_count >= len(is_above) / 2):
        if((record[11] == ">50K")):
           predicted_correct += 1
    else:
        if((record[11] == "<=50K")):
           predicted_correct += 1
    total_tests += 1
    return total_tests, predicted_correct

def print_results(total_tests, predicted_correct):
    print("\nOUTCOME\n")
    print("Records Compared:", total_tests)
    print("Correct Predictions:", predicted_correct)
    accuracy = (predicted_correct / total_tests) * 100
    print("Accuracy:", accuracy, "%")
    print("By Cian O'Gorman TCD")

def main():
    cleaned_dataset = get_clean_data(DATA_URL)

    training_dataset = cleaned_dataset[:int(len(cleaned_dataset) * TEST_TRIAL_SPLIT)]
    test_dataset = cleaned_dataset[int(len(cleaned_dataset) * TEST_TRIAL_SPLIT):]
    classifier_mid_points = create_classifier(training_dataset)
    total_tests, predicted_correct = test_classifier(test_dataset, classifier_mid_points)
    print_results(total_tests, predicted_correct)

if __name__ == "__main__":
    main()

