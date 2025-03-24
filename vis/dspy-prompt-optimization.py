import marimo

__generated_with = "0.11.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Prompt Tuning for Language Models from Basics to Mastery (Cont)
        <center>Sadamori Kojaku</center>

        /// warning | This notebook is a subsection of the main notebook. 🙃
        This notebook is a part of "Prompt Tuning for Language Models from Basics to Mastery" found at [prompt-tuning.py](https://static.marimo.app/static/prompt-tuning-o8po). 

        ///

        /// tip | How to run this notebook

        To run the notebook, first download it as a `.py` file, then use the following steps:

        Install **marimo**:
        ```bash
        pip install marimo
        ```

        Install **uv** (a Python package manager that automatically manages dependencies):
        ```bash
        pip install uv
        ```

        Launch the notebook
        ```bash
        marimo edit --sandbox <filename>.py
        ```

        The notebook will open in your web browser. All necessary packages will be installed automatically in a dedicated virtual environment managed by **uv**.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Set up

        ![](https://miro.medium.com/v2/resize:fit:1168/0*-9J6J-vqMDaHr174.png)

        We will use [ollama](https://ollama.com/) to run **a small language model** (SLM). Ollama is a lightweight runtime engine for running large language models locally. With ollama, you can chat language models running on your machine.

        To install ollama, see 👉[Windows](https://www.youtube.com/watch?v=3W-trR0ROUY), [Mac](https://www.metriccoders.com/post/how-to-install-and-run-ollama-on-macos), and [Linux](https://github.com/ollama/ollama/blob/main/docs/linux.md).  We will use `gemma3` which can be downloaded by

        ```bash
        ollama pull gemma3
        ```

        If you have a serve with more powerful GPU cards, you can install ollama on the server, and then ssh-tunneling it by:
        ```bash
        ssh -N -L 11434:localhost:11434 username@servername
        ```

        (If gemma3 is too slow, you can try 'phi3', which is faster but less accurate.)
        """
    )
    return


@app.cell
def _():
    MODEL_NAME = "gemma3"
    return (MODEL_NAME,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Optimizer: Prompt Optimization
        DSPy provides prompt optimization for language models, which automates the process of finding the best prompt for a given task.

        To showcase, let us consider a service ticket classification task. The dataset is taken from  [Customer IT Support Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download), which is a collection of customer support tickets, including, the subject, body, and queue, and priority.
        """
    )
    return


@app.cell
def _(MODEL_NAME):
    import dspy
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from dspy.teleprompt import COPRO, MIPROv2
    import litellm
    litellm.drop_params = True

    lm = dspy.LM(f'ollama_chat/{MODEL_NAME}', api_base='http://localhost:11434', api_key='')
    dspy.settings.configure(lm=lm)
    return COPRO, MIPROv2, dspy, litellm, lm, np, pd, train_test_split


@app.cell
def _(pd):
    data_table = pd.read_csv("https://raw.githubusercontent.com/skojaku/applied-soft-comp/refs/heads/master/data/service-ticket-dataset/data.csv")
    data_table = data_table.sample(50)
    return (data_table,)


@app.cell
def _(data_table):
    data_table.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We put the data into DSPy-compatible data format.""")
    return


@app.cell
def _(data_table, dspy, train_test_split):
    ticket_data = []
    for i, datum in data_table.iterrows():
        example = dspy.Example(
            body=datum['body'],
            subject=datum['subject'],
            queue=datum['queue'],
            priority=datum['priority']
        ).with_inputs("body", "subject")  # Specify that 'message' is an input field
        ticket_data.append(example)

    # Split the data into training and evaluation sets
    train_data, eval_data = train_test_split(ticket_data, test_size=0.5, random_state=42)
    return datum, eval_data, example, i, ticket_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our goal is to create a prompt that can classify the service ticket into the correct queue. To start, let us define a signature for the classifier.""")
    return


@app.cell
def _(dspy):
    class ServiceTicketQueueClassifier(dspy.Signature):
        """Analyze a message to determine the service ticket type ('Billing and Payments', 'Customer Service', 'IT Support', 'Product Support', 'Technical Support')"""
        body = dspy.InputField(desc="The message to analyze")
        subject = dspy.InputField(desc="The subject of the message")
        queue = dspy.OutputField(desc="The type of service ticket")
        reasoning = dspy.OutputField(desc="Reasoning for the queue classification")
    return (ServiceTicketQueueClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then, create a module that uses the signature.""")
    return


@app.cell
def _(ServiceTicketQueueClassifier, dspy):
    # Create a two-stage classifier
    class ServiceTicketClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.queue_classifier = dspy.ChainOfThought(ServiceTicketQueueClassifier)

        def forward(self, body, subject, **params):
            queue_result = self.queue_classifier(body=body, subject=subject)
            return queue_result
    return (ServiceTicketClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In the last step, we define the evaluation metrics that guide the optimization process.""")
    return


@app.cell
def _():
    def evaluate_classification(example, prediction, trace=None):
        if example.queue == prediction.queue:
            return 1.0
        else:
            return 0.0
    return (evaluate_classification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let us now optimize the model. DSPy supports different prompt optimization strategies.

        1. **Few-shot prompting**: DSPy will generate a few examples and use them to optimize the prompt.
        2. **Instruction prompt optimization**: DSPy will generate a new instruction for the model
        3. **Parameter-finetuning**: DSPY will finetune the parameters of the model

        For illustration purposes, let us restrict ourselves to a simple instruction optimization based on the `COPRO` optimizer. For more details, please refer to the [DSPy documentation](https://dspy.ai/learn/optimization/optimizers/).
        """
    )
    return


@app.cell
def _(COPRO, ServiceTicketClassifier, evaluate_classification, train_data):
    # Set up the optimizer
    _optimizer = COPRO(
        metric=evaluate_classification,
        auto="light", # Can choose between light, medium, and heavy optimization runs
    )

    # Create the model
    clf_model = ServiceTicketClassifier()

    # Run optimization (note: this can be computationally intensive with local models)
    optimized_clf_model = _optimizer.compile(
        student=clf_model,
        trainset=train_data,
        eval_kwargs=dict(num_threads=4, display_progress=False, display_table=0)
    )
    return clf_model, optimized_clf_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now the tuning is done. Let us evaluate the performance of the optimized model.""")
    return


@app.cell
def _(clf_model, optimized_clf_model, train_data):
    def evaluate_model(model, dataset):
        queue_accuracy = 0
        for example in dataset:
            prediction = model(body=example.body, subject=example.subject)
            if example.queue == prediction.queue:
                queue_accuracy += 1
        return queue_accuracy / len(dataset)

    _accuracy_original = evaluate_model(clf_model, train_data)
    _accuracy_optimized = evaluate_model(optimized_clf_model, train_data)
    print(f"Accuracy: {_accuracy_original:.2f} (original) -> {_accuracy_optimized:.2f} (optimized)")
    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is the prompts before and after optimization.""")
    return


@app.cell
def _(optimized_clf_model, textwrap):
    print(textwrap.fill(optimized_clf_model.queue_classifier.predict.signature.instructions))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, let us toggle the optimization via few-shot prompting, along with instruction optimization. This can be done by using the `MIPROv2` optimizer.""")
    return


@app.cell
def _(MIPROv2, ServiceTicketClassifier, evaluate_classification, train_data):
    _optimizer = MIPROv2(
        metric=evaluate_classification,
        auto="light",
        num_threads=4,
        verbose=False,
    )

    clf_model_fewshot = ServiceTicketClassifier()

    optimized_model_fewshot = _optimizer.compile(
        student=clf_model_fewshot,
        trainset=train_data,
        max_bootstrapped_demos=3, # Number of maximum bootstrapped demonstrations
        max_labeled_demos=3, # Number of maximum labeled demonstrations
        requires_permission_to_run=False,
    )
    return clf_model_fewshot, optimized_model_fewshot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Bootstrapped demonstrations are the examples generated by the model. The labeled demonstrations are the examples in the training data. You can also use a powerful `teacher` model to generate the demonstrations and labels. See the [DSPy documentation](https://dspy.ai/learn/optimization/optimizers/) for more details.""")
    return


@app.cell
def _(clf_model, evaluate_model, optimized_model_fewshot, train_data):
    _accuracy_original = evaluate_model(clf_model, train_data)
    _accuracy_optimized = evaluate_model(optimized_model_fewshot, train_data)
    print(f"Accuracy: {_accuracy_original:.2f} (original) -> {_accuracy_optimized:.2f} (optimized)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is the few-shot examples generated by the model and the optimized instruction.""")
    return


@app.cell
def _(optimized_model_fewshot, textwrap):
    print("Few-shot examples:")
    for demo in optimized_model_fewshot.queue_classifier.predict.demos:
        print(demo)
    print("-------")
    print("Optimized instruction:")
    print(textwrap.fill(optimized_model_fewshot.queue_classifier.predict.signature.instructions))
    return (demo,)


@app.cell
def _():
    import marimo as mo
    import textwrap
    return mo, textwrap


if __name__ == "__main__":
    app.run()
