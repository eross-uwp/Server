def predict(original_data):
    sum_of_gpa = 0
    for row in original_data:
        sum_of_gpa = sum_of_gpa + row
    prediction = sum_of_gpa/original_data.size
    return prediction
