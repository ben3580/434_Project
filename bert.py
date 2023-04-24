from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, create_optimizer, TFAutoModelForQuestionAnswering, pipeline 
import tensorflow as tf
from keras import backend as K

# Load SQuAD v2 dataset
squad = load_dataset("squad_v2", split="train")
squad = squad.train_test_split(test_size=0.2)

# Import pretrainied tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess function for dataset
# Find the start and end locations of the answer to each question
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = 0
        end_char = 0
        if len(answer["answer_start"]) > 0:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            if start_char == end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

data_collator = DefaultDataCollator(return_tensors="tf")

# Hyperparams
batch_size = 16
num_epochs = 10
total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)

# Use BERT model
model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Define training and testing dataset
tf_train_set = model.prepare_tf_dataset(
    tokenized_squad["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_squad["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

# Compile the model
model.compile(optimizer=optimizer, metrics=['acc'])

# Train and evaluate
for i in range(num_epochs):
    model.fit(x=tf_train_set, epochs=1, verbose=2)
    model.evaluate(x=tf_validation_set, verbose=2)

# Q-A visualization
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
    