from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, create_optimizer, AutoModelForQuestionAnswering, pipeline 
import tensorflow as tf
from keras import backend as K

model_name = "deepset/tinyroberta-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)
print(res)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="tf")

context = """Computational complexity theory is a branch of the theory of computation in theoretical computer science\
that focuses on classifying computational problems according to their inherent difficulty, and relating\
those classes to each other. A computational problem is understood to be a task that is in principle\
amenable to being solved by a computer, which is equivalent to stating that the problem may be solved\
by mechanical application of mathematical steps, such as an algorithm.
"""
question1 = """What branch of theoretical computer science deals with broadly classifying computational problems by difficulty and class of relationship?"""
question2 = """By what main attribute are computational problems classified utilizing computational complexity theory?"""
question3 = """What is the term for a task that generally lends itself to being solved by a computer?"""
question4 = """What is computational complexity principle?"""
question5 = """What branch of theoretical computer class deals with broadly classifying computational problems by difficulty and class of relationship?"""
question6 = """What is understood to be a task that is in principle not amendable to being solved by a computer?"""
question7 = """What cannot be solved by mechanical application of mathematical steps?"""
question8 = """What is a manual application of mathematical steps?"""

questions = [question1, question2, question3, question4, question5, question6, question7, question8]

for q in questions:
    print(question_answerer(question=q, context=context))
    