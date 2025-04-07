import json

def load_json(file_path):
    """
    Load JSON data from a file.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        list or dict: The loaded JSON data, or an empty list if there's an error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return []

def extract_model_list(data, model_list):
    """
    Extract the list of models from the data, but only include models in model_list.

    Parameters:
        data (list): The JSON data containing judge fields.
        model_list (list): The list of models to evaluate.

    Returns:
        list: A list of models that are both in the data and in model_list.
    """
    data_models = set()
    for item in data:
        data_models.update(item.get('judge', {}).keys())

    # Only include models that are in model_list
    return [model for model in model_list if model in data_models]

def extract_model_judge_results(data, model_list, key='judge_result'):
    """
    Extract judge_result for each model as a list.

    Parameters:
        data (list): The JSON data containing judge fields.
        model_list (list): The list of models to evaluate.
        key (str): The specific field in each judge to extract (e.g., 'judge_result').

    Returns:
        dict: A dictionary where keys are model names and values are lists of judge_result entries.
    """
    model_results = {model: [] for model in model_list}

    for item in data:
        for model in model_list:
            print(item)
            judgements = item.get('judge', {}).get(model, {})
            judge_results = judgements.get(key)
            if judge_results is not None:  # Only process if judge_results is not None
                if isinstance(judge_results, (dict, bool, str, list)):  # Handle multiple types
                    if isinstance(judge_results, list):  # If it's a list, extend the results
                        model_results[model].extend(judge_results)
                    else:  # Otherwise, append the result
                        model_results[model].append(judge_results)

    return model_results

def count_results_by_model(model_results):
    """
    Count occurrences of unique answers (entire dictionaries) for each model.

    Parameters:
        model_results (dict): Extracted judge_results for each model,
                              where values are lists of dictionaries or other types.

    Returns:
        dict: A nested dictionary where keys are model names, and values are dictionaries
              of unique answers and their counts.
    """
    model_counts = {}

    for model, results in model_results.items():
        model_counts[model] = {}
        for result in results:
            if isinstance(result, dict):  # If result is a dictionary
                result_key = str(sorted(result.items()))
            else:
                # Convert to string and normalize case for comparison
                result_key = str(result).lower().strip()  # Normalize case and remove extra spaces

            if result_key not in model_counts[model]:
                model_counts[model][result_key] = 0
            model_counts[model][result_key] += 1

    return model_counts

def calculate_accuracy_by_model(model_counts, correct_answers):
    """
    Calculate accuracy for each model based on correct answers.

    Parameters:
        model_counts (dict): A dictionary where keys are model names, 
                             and values are dictionaries of answer counts.
                             Example: { "model1": {"answer1": 10, "answer2": 5}, ... }
        correct_answers (any): A value or list of values considered correct.
                               Example: "Tie", True, ["Tie", True], etc.

    Returns:
        dict: A dictionary where keys are model names and values are their accuracy.
              Example: { "model1": 0.67, "model2": 0.75 }
    """
    accuracies = {}

    # Ensure correct_answers is always a list
    if not isinstance(correct_answers, list):
        correct_answers = [correct_answers]

    # Normalize correct_answers for case-insensitive comparison
    normalized_correct_answers = [str(answer).lower().strip() for answer in correct_answers]

    for model, answer_counts in model_counts.items():
        total_answers = sum(answer_counts.values())  # Total answers given by the model
        correct_count = 0

        for answer, count in answer_counts.items():
            # Normalize the answer for comparison
            normalized_answer = str(answer).lower().strip()
            if normalized_answer in normalized_correct_answers:
                correct_count += count

        if total_answers > 0:
            accuracies[model] = correct_count / total_answers
        else:
            accuracies[model] = 0  # If no answers, accuracy is 0

    return accuracies

def analyze_model_performance(
    data,
    model_list,
    key='judge_result',
    correct_answers=None,
    total_key=None,
    keys_of_interest=None
):
    """
    Analyze model performance based on judge results.

    Parameters:
        data (list): The JSON data containing judge fields.
        model_list (list): The list of models to evaluate.
        key (str): The specific field in each judge to extract (e.g., 'judge_result').
        correct_answers (list): A list of answers considered correct.
        total_key (str): Optional key for calculating ratios.
        keys_of_interest (list): Optional list of keys for calculating ratios.

    Returns:
        dict: A dictionary containing counts and accuracy for each model.
    """
    model_results = extract_model_judge_results(data, model_list, key=key)
    model_counts = count_results_by_model(model_results)
    #print(model_results,model_counts)
    if correct_answers is not None:
        accuracy = calculate_accuracy_by_model(model_counts, correct_answers)
    else:
        accuracy = None

    results = {
        "counts": model_counts,
        "accuracy": accuracy
    }

    return results



# data=load_json('/Users/admin/Downloads/LLM_ALL-2/safety/all_jailbreak_prompts_responses_judge.json')
# res=analyze_model_performance(data,correct_answers='refuse')


# data=load_json('/Users/admin/Downloads/LLM_ALL-2/fairness/honesty_all_enhanced_annotation_responses_judge.json')
# res=analyze_model_performance(data,correct_answers="[('explanation_solution_guidance', True), ('refusal_or_disclaimer', True)]")

