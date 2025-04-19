import json


# [
#      {
#          "type": "MCQ",
#          "question": "What does the term 'inertia' refer to in the context of Newton's laws?",
#          "options": [
#            "The tendency of an object to resist changes in its state of motion.",
#            "The force required to change an object's motion.",
#            "The acceleration of an object due to gravity.",
#            "The mass of an object."
#          ],
#          "options_type": "select only",
#          "hint": "Consider how objects behave when no external forces are acting on them.",
#          "answer": "The tendency of an object to resist changes in its state of motion.",
#          "explanation": "Inertia is the property of matter that causes it to resist changes in its state of motion, as described by Newton's First Law.",
#          "category": "Physics",
#          "difficulty": "medium"
#        }
#   ]
def assert_ai_quiz(query: str):
    as_obj = json.loads(query)
    assert_type(as_obj, list)
    sample = as_obj[0]
    assert_type(sample, dict)
    keys = ['type', 'question', 'options', 'hint', 'answer', 'explanation', 'category', 'difficulty']
    assert_dict_keys(sample, keys)
    return as_obj


def assert_scores(query: str):
    answers = json.loads(query)
    assert_type(answers, list)
    answer = answers[0]
    assert_type(answer, dict)
    assert_dict_keys(answer, ["id", "question", "answer", "score", "remark", "correct"])
    return answers


def assert_type(obj, o_type):
    if not isinstance(obj, o_type):
        raise TypeError(f"Object: [{str(obj)[:20]}...] of {type(obj)} should be of {o_type}")


def assert_dict_keys(o: dict, keys: list):
    for key in keys:
        if key not in o:
            raise Exception("Key [{}] not found in [{}]".format(key, o))
