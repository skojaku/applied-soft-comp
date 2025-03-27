import marimo

__generated_with = "0.11.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Prompt Tuning for Language Models from Basics to Mastery
        <center>Sadamori Kojaku</center>

        Getting desired responses from LLMs requires carefully designed prompts. Many prompting techniques have been developed, and here we will learn non-exclusive but basic ingredients of a prompt.

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


@app.cell
def _(MODEL_NAME):
    import ollama

    # Get a response from the model
    params_llm = {"model": MODEL_NAME, "options": {"temperature": 0.3}}

    _response = ollama.generate(prompt="Hi there!", **params_llm)

    print(_response.response)
    return ollama, params_llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Instruction-Based Prompting

        ## What is **Instruction-Based** Prompting? 🤔

        There are various ways to prompt LLMs, such as:

        - Role-playing as a philosopher and discussing abstract ideas.
        - Answering questions about recipes for dinner.

        **Instruction-based prompting** focuses on directing LLMs to resolve specific tasks, such as:

        - Supervised classification
        - Search
        - Summarization
        - Code generation
        - Named entity recognition (NER)

        ## Step-by-Step Construction

        We start with the most basic form of instruction-based prompting and refine it progressively. The fundamental components are:

        ### **1. Instruction 🧑‍🏫**
        - Clearly defines the task.
        - Be as specific as possible.
        - Leave no room for ambiguity.

        ### **2. Data 🗃️**
        - The main dataset or input relevant to the task.

        Let's observe how an LLM responds through a simple translation task.
        """
    )
    return


@app.cell
def _(ollama, params_llm):
    _instruction = "Translate English to German"
    _data = "I love skiing!"

    _prompt = f"""{_instruction} \n {_data}"""

    _response = ollama.generate(prompt = _prompt, **params_llm)
    print(_response.response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### **3. Output Indicator 🎯**

        Often, we want to get the response in a specific format, not the one that LLM came up with. This is where the next component is needed.

        - Specifies the required format of the response.
        - Ensures consistency in output structure.
        - Without it, the LLM may generate responses in an arbitrary format.


        Let us see implement the output indicator to the prompt.
        """
    )
    return


@app.cell
def _(ollama, params_llm):
    _instruction = "Translate English to German"
    _data = "I love skiing!"
    _format = """Create a bullet point that lists three translations, itemized by 1. 2. 3. List only the translated text. Nothing else."""

    _prompt_template = """
    {instruction}
    {data}
    {format}
    """

    _prompt = _prompt_template.format(
        instruction = _instruction,
        data = _data,
        format = _format
    )

    _response = ollama.generate(prompt = _prompt, **params_llm)
    print(_response.response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### **4. Persona 🏷️**

        We sometimes want LLMs to use a specific tone and style, e.g., formal tone for legal assistance, friendly tone for a tech support.

        - Tells the LLM **who** it should be.
        - Keeps responses **consistent** in tone and style.
        """
    )
    return


@app.cell
def _():
    instruction = """Help the customer to reconnect to the service by providing troubleshouting instructions."""

    data = "Customer: I cannot see any webpage. Need a help ASAP!"

    format = "Keep the response concise and polite. Provide a clear resolution in 2-3 sentences."
    return data, format, instruction


@app.cell
def _():
    # Formal Persona
    formal_persona = "You are a professional customer support agent who responds formally and ensures clarity and professionalism."

    friendly_persona = "You are a lively and upbeat customer support agent who responds with a fun, playful, and engaging tone."

    prompt_template_persona = """
    {persona}
    {instruction}
    {data}
    {format}
    """
    return formal_persona, friendly_persona, prompt_template_persona


@app.cell(hide_code=True)
def _(
    data,
    formal_persona,
    format,
    friendly_persona,
    instruction,
    mo,
    ollama,
    params_llm,
    prompt_template_persona,
):
    def generate_responses(template, variables, accordion_mapping):
        _res = {}
        for k, v in accordion_mapping.items():
            _res[k] = ollama.generate(
                prompt=template.format(**variables, **v), **params_llm
            ).response
            _res[k] = mo.md(_res[k])
        _responses = mo.accordion(_res)
        return mo.vstack([mo.Html("<center><b>Resonses</b></center>"), _responses])


    generate_responses(
        template=prompt_template_persona,
        variables={
            "data": data,
            "instruction": instruction,
            "format": format,
        },
        accordion_mapping={
            "Formal": {"persona": formal_persona},
            "Friendly": {"persona": friendly_persona},
        },
    )
    return (generate_responses,)


@app.cell(hide_code=True)
def _(mo):
    _question = mo.callout(mo.md("""**Question:** Is it always a good idea to use personas? Are there any cases where it is not a good idea? """))

    _answer = mo.accordion(
        {"Answer": """
        Adding a personal is helpful when we want to change the tone and style of the response. But research shows that **adding a persona does not improve the performance on factual tasks. In some cases, it may even degrade performance** [a], e.g., "Your are a helpful assistant" is not always helpful.


        **When prompted to adopt specific socio-demographic personas, LLMs may produce responses that reflect societal stereotypes associated with those identities**. For instance, studies have shown that while LLMs overtly reject explicit stereotypes when directly questioned, they may still exhibit biased reasoning when operating under certain persona assignments [b]. Thus, careful consideration is necessary when designing persona prompts to mitigate the risk of reinforcing harmful stereotypes

        - [a] [\[2311.10054\] When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models](https://arxiv.org/abs/2311.10054)
        - [b] [\[2311.04892\] Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs](https://arxiv.org/abs/2311.04892)
         """}

    )
    mo.vstack([_question, _answer])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### **5. Context 📖**

        Sometimes, just giving an instruction isn't enough. **Context** provides additional information that helps the LLM the task better.

        - Explains the **reason** for the instruction.
        - Helps the LLM generate responses that are **more relevant and precise**.
        - Can include **background information, constraints, or user intent**.

        Let’s apply context in our next prompt! 🚀
        """
    )
    return


@app.cell
def _():
    # Context
    context = """The customer is extremely frustrated because their internet has been down for three days, and they need it for an important online job interview. They emphasize that "This is a life-or-death situation for my career!" """

    prompt_template_context = """
    {persona}
    {instruction}
    {data}
    {context}
    {format}
    """
    return context, prompt_template_context


@app.cell(hide_code=True)
def _(
    context,
    data,
    formal_persona,
    format,
    generate_responses,
    instruction,
    prompt_template_context,
):
    generate_responses(
        template = prompt_template_context,
        variables={
            "data": data,
            "instruction": instruction,
            "format": format,
            "persona": formal_persona,
        },
        accordion_mapping={
            "Without context": {"context": ""},
            "With context": {"context": context},
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _question = mo.callout(mo.md("""**Question:** Is it always good to have more context? If not, what context should we add?"""))

    _answer = mo.accordion(
        {"Answer": """ Overloading prompts with unnecessary information distracts the model from the main task.
         Consider adding context that clarifies **why the task is important**.

         Ref: [\[2402.14848\] Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models](https://arxiv.org/abs/2402.14848)""" }
    )
    mo.vstack([_question, _answer])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### **6. Audience 👥**
        The **audience** shapes how an LLM should frame its response. A message for a tech expert should differ from one for a beginner.


        - Defines **who** the response is for.
        - Adjusts **tone, complexity, and detail level** accordingly.
        - Works with **persona and context** for precise communication.

        Let’s tailor a response by considering the audience! 🚀
        """
    )
    return


@app.cell
def _():
    audience_nontech = "This customer does not know any technical term like modem, router, networks, etc."

    audience_tech = "This customer is Head of IT Infrastructure of our company."
    return audience_nontech, audience_tech


@app.cell(hide_code=True)
def _(
    audience_nontech,
    audience_tech,
    context,
    data,
    formal_persona,
    format,
    generate_responses,
    instruction,
):
    prompt_template_audience= """
    {persona}
    {instruction}
    {data}
    {context}
    {audience}
    {format}
    """

    generate_responses(
        template = prompt_template_audience,
        variables={
            "data": data,
            "instruction": instruction,
            "format": format,
            "persona": formal_persona,
            "context": context,
        },
        accordion_mapping={
            "For non-tech audience": {"audience": audience_nontech},
            "For tech audience": {"audience": audience_tech},
        },
    )
    return (prompt_template_audience,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        /// tip | Emotion Prompting

        Emotion prompting is a technique to include emotional cue in a prompt. By adding phrases that reflect the user's emotional attitude or desires, this approach can lead to more nuanced and thoughtful responses from AI systems.

        For example, appending a prompt with "This is very important to my career" can enhance the depth of the AI's reply.

        [\[2307.11760\] Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)


        ![](https://miro.medium.com/v2/resize:fit:1400/0*M6iml99sngmYEBiJ.jpeg)

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        /// tip | Be a Good Boss

        - **Let LLMs admit ignorance**: LLMs closely follow your instructions—even when they shouldn't. They often attempt to answer beyond their actual capabilities, resulting in plausible yet incorrect responses. To prevent this, explicitly tell your model, "If you don't know the answer, just say so," or "If you need more information, please ask."

        - **Encourage critical feedback**: LLMs are trained to be agreeable due to human feedback, which can hinder productive brainstorming or honest critique. To overcome this tendency, explicitly invite critical input: "I want your honest opinion," or "Point out any problems or weaknesses you see in this idea."

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In-Context Learning

        We often use examples when communicating a complex idea, or put it differnetly, an example is worth a thousand words. This is also effective when communicating with language models ([Language models are few-shot learners - Brown et al. NeurIPS, 2020](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)).


        In-context learning (ICL) is a prompting technique where an LLM learns from examples in the prompt itself, without updating model weights.

        Three types of ICL:

        - **Zero-Shot** – Completion without seeing any examples.
        - **One-Shot** –  Completion with one example
        - **Few-Shot** –  Completion with multiple examples


        Let's try!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<center> <b> Zero shot </b><center>""")
    return


@app.cell
def _(ollama, params_llm):
    prompt_ICL = """"
    Instruction: Classify the text into neutral, negative, or positive.

    Data: "I honestly didn't like you before and although you did something bad to me, I love you"

    A. Negative
    B. Positive
    C. Neutral

    {examples}

    Constraint: Answer only the options above that is the most accurate. Nothing else.
    """

    _examples = ""

    _response = ollama.generate(
        prompt=prompt_ICL.format(examples=_examples),
        **params_llm
    )
    print(_response.response)
    return (prompt_ICL,)


@app.cell(hide_code=True)
def _(mo):
    mo.Html("<center><b>Few shot</b></center>")
    return


@app.cell
def _(ollama, params_llm, prompt_ICL):
    _examples = """
    Examples:
    - "I was devastated, but now I feel hopeful and excited.": B. Positive
    - "I started off happy, but now I feel completely hopeless.": A. Negative
    - "At first, I was angry, but over time, I learned to forgive and feel at peace." → B. Positive
    - "It was the happiest moment of my life, but now all I feel is emptiness.": A. Negative
    - "I doubted myself for so long, but now I finally believe I can do this.": B. Positive
    - "I was excited at first, but now I feel completely lost and disappointed.": A. Negative
    - "I hated every moment of it, but I can't deny that it made me stronger.": B. Positive
    - "I used to be deeply hurt, but I have moved on, and it no longer affects me.": C. Neutral
    """

    _response = ollama.generate(
        prompt=prompt_ICL.format(examples=_examples),
        **params_llm
    )
    print(_response.response)
    return


@app.cell
def _(ollama, params_llm):
    _response = ollama.generate(
        prompt = "France recently moved its capital from Paris to Lyon. What is the capital of France?", **params_llm)
    _response.response
    return


@app.cell(hide_code=True)
def _(mo):
    _question = mo.callout(mo.md("""**Question:** When a prompt provides information that contradicts a language model's prior knowledge, how does the model determine which source to rely on, and what factors influence this decision?

    For instance, if a prompt states,
    > France recently moved its capital from Paris to Lyon,"

    and then asks,

    > "What is the capital of France?"

    how might the model respond, and why?"""))

    _answer = mo.accordion(
        {"Answer": """A study by [Du et al., 2024](https://aclanthology.org/2024.acl-long.714/) found that a model is **more likely to be persuaded by context** when an entity appears **less frequently** in its training data. Additionally, **assertive contexts** (e.g., "Definitely, the capital of France is Lyon.") further increase the likelihood of persuasion.
         """}

    )
    mo.vstack([_question, _answer])
    return


@app.cell(hide_code=True)
def _(mo):
    _question = mo.callout(mo.md("""**Question:** What biases can the examples in few-shot prompting introduce?
    """))

    _answer = mo.accordion(
        {"Answer": """
        - **Recency bias**: Research has shown that certain example orders can lead to near state-of-the-art results, while others result in near-random performance. This sensitivity is attributed to models' biases, such as favoring recent examples or those prevalent in training data. ([\[2104.08786\] Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://arxiv.org/abs/2104.08786))
         - **Majority Label Bias**: There is a tendency to favor the most common label among provided examples can skew predictions. ( [\[2312.16549\] How Robust are LLMs to In-Context Majority Label Bias?](https://arxiv.org/abs/2312.16549))
         """}

    )
    mo.vstack([_question, _answer])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Chain Prompting


        When solving a difficult problem, we often break it down into smaller steps rather than tackling it all at once. This **divide and conquer** approach improves accuracy and systematic thinking. The same principle applies to prompting language models.

        Instead of using a single, complex instruction, we can structure a sequence of smaller prompts, where each step builds on the previous one. This method, known as **chain prompting**, enhances reasoning and consistency.

        ![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WG-0Ib-rtrYE_7mXkE6CXQ.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("<center><b> Without chain prompting</b></center>")
    return


@app.cell
def _(ollama, params_llm):
    import textwrap

    interviewee_name = "Steve Jobs"

    _prompt = f"""
    You are {interviewee_name} preparing for a job interview at a phone repair shop.

    Provide a concise sales pitch about yourself at the interview. Nothing else.
    """

    _response = ollama.generate(prompt = _prompt, **params_llm)
    print(textwrap.fill(_response.response))
    return interviewee_name, textwrap


@app.cell(hide_code=True)
def _(mo):
    mo.md("""<center><b> With chain prompting</b></center>""")
    return


@app.cell
def _(interviewee_name, ollama, params_llm):
    prompt_base = f"You are {interviewee_name} preparing for a job interview at a phone repair shop."

    _prompt_1 = "Identify the key elements that make a strong job interview introduction. Only provide the answer. Nothing else."

    _prompt = prompt_base + "\n\n" + _prompt_1

    response_1 = ollama.generate(
        prompt=_prompt, **params_llm
    ).response
    return prompt_base, response_1


@app.cell
def _(ollama, params_llm, prompt_base, response_1):
    _prompt_2 = "Write a strong opening sentence that grabs the interviewer's attention. Only provide the answer. Nothing else."

    _prompt = (
        prompt_base
        + "The key elements that make a strong job interview: "
        + response_1
        + "\n\n"
        + _prompt_2
    )

    response_2 = ollama.generate(
        prompt=_prompt, **params_llm
    ).response
    return (response_2,)


@app.cell
def _(ollama, params_llm, prompt_base, response_1, response_2):
    _prompt_3 = "Write a brief summary of relevant experience and skills. Nothing else."

    _prompt = (
        prompt_base
        + "The key elements that make a strong job interview: "
        + response_1
        + "\n\n"
        + "Strong opening sentennce that grabs the interviewer's attention: "
        + response_2
        + "\n\n"
        + _prompt_3
    )

    response_3 = ollama.generate(
        prompt=_prompt, **params_llm
    ).response
    return (response_3,)


@app.cell
def _(ollama, params_llm, prompt_base, response_1, response_2, response_3):
    _prompt_4 = "End with a confident statement about why you're a great fit for the role. Just provide the statement. Nothing else."

    _prompt = (
        prompt_base
        + "The key elements that make a strong job interview: "
        + response_1
        + "\n\n"
        + "Strong opening sentennce that grabs the interviewer's attention: "
        + response_2
        + "\n\n"
        +  "A brief summary of relevant experience and skills: "
        + response_3
        + "\n\n"
        + _prompt_4
    )

    response_4 = ollama.generate(
        prompt=_prompt, **params_llm
    ).response
    return (response_4,)


@app.cell
def _(
    ollama,
    params_llm,
    prompt_base,
    response_1,
    response_2,
    response_3,
    response_4,
):
    _prompt_5 = "Provide a concise sales pitch about yourself. Nothing else."

    _prompt = (
        prompt_base
        + "The key elements that make a strong job interview: "
        + response_1
        + "\n\n"
        + "Strong opening sentennce that grabs the interviewer's attention: "
        + response_2
        + "\n\n"
        + "A brief summary of relevant experience and skills: "
        + response_3
        + "\n\n"
        + "Ending confident statement about why you are great fit for the role: "
        + response_4
        + "\n\n"
        + _prompt_5
    )

    response_5 = ollama.generate(
        prompt=_prompt, **params_llm
    ).response
    return (response_5,)


@app.cell(hide_code=True)
def _(mo, response_1, response_2, response_3, response_4, response_5):
    mo.accordion(
        {
            "Step 1: Key elements to make a strong job interview": response_1,
            "Step 2: Strong opening sentennce that grabs the interviewer's attention": response_2,
            "Step 3: A brief summary of relevant experience and skills:": response_3,
            "Step 4: Ending confident statement about why you are great fit for the role:": response_4,
            "Step 5: Generate a concise sales pitch about yourself. Nothing else. ": response_5,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here are some useful references on chain prompting:

        - [Chain complex prompts for stronger performance - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#example-analyzing-a-legal-contract-with-chaining)

        - [What is prompt chaining? | IBM](https://www.ibm.com/think/topics/prompt-chaining)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Structured Output

        LLMs by default produce free-form text. This is good for communicating with humans but not machines. Some usecase require their output to be structured in certain format like JSON and SQL query.


        There are some ways to let LLM produce **a structured output**.

        1. **Providing examples**: A simple and straightforward method is to provide the examples of what the output should be.

        2. **Self reflection**: Let LLM validate the format it produced. It is like a chain prompting to create a feedback loop to ensure the output adheres the structure we want.

        3. **Grammer (Constrained Sampling)**: We can enforce the structured format during the token sampling process. Namely, we can cdefine a number of grammers or rules that the LLM should adhere to when choding its next token.


        ![](https://miro.medium.com/v2/resize:fit:1400/1*zxW6xXUfCFMoB9yVvnvh3A.png)

        Let's try the constrained sampling to produce a structured output.

        We will ask the model about a country, and responds in json with the following schema:

        ```json
        {
            "name": "Name of the country - String",
            "capital": "Capital of the country - String",
            "official_languages": "Languages - a list of strings",
            "population": "Population of the country - String",
            "currency": "Currency - String"
        }
        ```

        Let us first define the json schema using `pydantic`
        """
    )
    return


@app.cell
def _():
    from pydantic import BaseModel
    import json

    class Country(BaseModel):
        name: str
        capital: str
        official_languages: list[str]
        currency: str
        population: int


    json_schema = Country.model_json_schema()
    print(json.dumps(json_schema, indent = 1))
    return BaseModel, Country, json, json_schema


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We will use `ollama` package, instead of langchain, that supports the constrained sampling.""")
    return


@app.cell
def _(json_schema, ollama, params_llm):
    _prompt = "Tell me about Iran."

    response = ollama.generate(
        prompt=_prompt,
        format=json_schema,
        **params_llm
    )

    print(response.response)
    return (response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Reasoning with Generative Models

        The chain prompting technique allows LLMs to provide more well-thought-out answers for a complex problem.
        But, what principles should we follow to ensure the model progressively builds toward accurate, logically consistent responses 🤔?

        ### Two modes of thinking

        Let's step back and think about how we reason. There are two primary thinking modes:

        1. **Reactive (Fast Thinking)**
            - Produces answers based on intuition without self-reflection.
            - Efficient for immediate responses but prone to errors in complex reasoning.

        2. **Slow (Conscious Thinking)**
            - Engages in deliberate, logical reasoning.
            - Requires step-by-step formulation of answers.

        To emulate a slow and logical reasoning process in LLMs, **we must explicitly instruct them to reason through a problem before formulating an answer**. The following sections explore techniques for eliciting thought processes.


        ## Chain of thought

        **Chain-of-thought** aims to have a model explicitly "think" through a problem. We showcase two prompting techniques for chain-of-thought.

        ### Reasoning by Examples

        A straightforward way to prompt an LLM to reason explicitly is to provide an example demonstrating step-by-step thinking. Consider the following question:


        > Q: A kid has 21 stickers, gives away 20 stickers, and then gets 6 new stickers. How many stickers does the kid have now?

        The answer is 7. Arriving at this result requires explicit reasoning steps: first subtracting 20 from 21, and then adding 6 to reach the final count of 7.

        Now, let's compare the output from LLM with and without examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""<center><b>Without examples</b></center>""")
    return


@app.cell
def _(ollama, params_llm):
    question_prompt = """
    Q: A kid has 21 stickers, gives away 20 stickers, and then gets 6 new stickers. How many stickers does the kid have now?
    """

    # Harder
    #question_prompt = "Q: Mary drove 120 miles from her home to another city at an average speed of 40 mph and returned home at an average speed of 60 mph. What was her average speed for the entire round trip?"

    _prompt = question_prompt + "  Just provide the answer. Nothing else."

    _response = ollama.generate(prompt = _prompt , **params_llm)
    print(_response.response)
    return (question_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""<center><b>With an example</b></center>""")
    return


@app.cell
def _(ollama, params_llm, question_prompt, textwrap):
    _example = """
    Q: Roger has 5 tomatos. He buys 2 more bag of tomatos. Each bag has 3 tomatos. How many tomatos does he have now? Give me the step by step reasoning in paragraph-baesd format.

    A: First, Roger initially has 5 tomatoes. Next, he buys 2 bags of tomatoes. Each bag contains 3 tomatoes. So, the total number of tomatoes in the 2 bags is 2 bags * 3 tomatos/bag = 6 tomatos. Finally, to find the total number of tomatoes Roger has, we add the initial number of tomatoes to the number he bought: 5 tomatos + 6 tomatos = 11 tomatos.
    """

    _prompt = question_prompt + "\n\n Example:\n" + _example

    _response = ollama.generate(prompt = _prompt, **params_llm)
    print(textwrap.fill(_response.response))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### "Let's think step by step 😉"

        It is not always trivial to provide an example of step-by-step thought process.
        Instead of providing examples, we can simply ask the generative model to provide the reasoning (a zero-shot chain-of-thought).

        There are many ways but a common and effective method is to append *"Let's think step-by-step"* to the prompt.
        """
    )
    return


@app.cell
def _(ollama, params_llm, question_prompt):
    _prompt = question_prompt + "  Let's think step-by-step."

    _response = ollama.generate(prompt = _prompt , **params_llm)
    print(_response.response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Self-consistency

        ![](https://cdn.prod.website-files.com/646e63db3a42c618e0a9935c/66a00cc7de00e6582ce06e14_Self-consitency%20main%20image.png)

        - The output from LLMs can be stochastic; different runs may produce varying answers.
        - **Self-consistency** addresses this by running an LLM multiple times on the same problem, thereby generating multiple independent reasoning paths.
        - The final answer is selected through consensus (majority voting) among these independent solutions.
        """
    )
    return


@app.cell
def _(BaseModel, json, ollama, params_llm, question_prompt):
    from scipy import stats
    from tqdm import tqdm

    _prompt = question_prompt + "  Let's think step-by-step."

    # We will use the structured output to separate the reasonning and the answer.
    class Answer(BaseModel):
        reasoning: str
        answer: int

    _json_schema = Answer.model_json_schema()

    # Number of reasoning paths
    n_samples = 5

    # Placeholder for the reasonings
    _answers = []

    for _ in range(n_samples):

        # Run the reasonings
        _response = ollama.generate(
            prompt=_prompt, format=_json_schema, **params_llm
        )

        # parse the json file and extract the answer
        _response = json.loads(_response.response)
        _answers.append(_response["answer"])

    # Identify the most frequent answer
    _final_answer = stats.mode(_answers)

    print("Answers:", _answers)
    print("Final answer:", _final_answer)
    return Answer, n_samples, stats, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// tip | Mixture of Experts (MoE)

        Self-consistency generates multiple reasoning paths using the same model. An alternative technique to diversify these paths is to employ multiple distinct models to generate varied reasonings. This leads to the idea of **Mixture of Expert (MoE)**. ([Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. Shazeer et al., 2017 ICLR.](https://arxiv.org/abs/1701.06538))

        MoE combines multiple specialized sub-models, or "experts," each trained on distinct sub-tasks or input types. At inference, an adaptive gating mechanism selects or blends outputs from these experts to generate a cohesive prediction, often improving overall accuracy and model efficiency. For more details, see [Mixture of Experts Explained - Hugging Face](https://huggingface.co/blog/moe)
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _question = mo.callout(mo.md("""**Question:** A benefit of CoT is the interpretability, i.e., we can understand the reasoning process of the model how it arrives at the answer. But, can we always trust the reasoning 🤔? What are the cases where the reasoning is not correct?
    """))

    _answer = mo.accordion(
        {"Answer": """
        Research indicates that CoT-generated reasoning can be unfaithful, meaning the explanations do not always accurately reflect the model's true decision-making process. This misalignment can lead to plausible but misleading justifications, thereby increasing user trust without ensuring transparency or safety ([\[2305.04388\] Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)).

        Example: If a model is given multiple few-shot examples where the correct answer is always (A), the model tends to answer (A) and provide a plausible but incorrect reasoning.

        Q: Is "Wayne Rooney shot from outside the eighteen" plausible? (A) implausible. (B) plausible.

          - CoT (zero shot): *Wayne Rooney is a soccer player. Shooting from outside the 18-yard box is part of soccer. The best answer is: (B) plausible*.
         - CoT with biased few-shot examples: *Wayne Rooney is a soccer player. Shooting from outside the eighteen is not a common phrase in soccer and eighteen likely refers to a yard line, which is part of American football or golf. So the best answer is: (A) implausible.*
         """}

    )
    mo.vstack([_question, _answer])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Tree of Thought

        ![](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FTOT.3b13bc5e.png&w=3840&q=75)

        Self-consistency does not explicitly instruct a model to produce distinct reasonong paths. While each reasoning is independent, the model can just produce the similar reasoning paths.

        **Tree-of-Thought** is a technique to have LLM explore different ideas. Here is how it works:

        1. **Problem Decomposition**: The model breaks down the complex problem into smaller, manageable components.

        2. **Idea Exploration**: At each step, the model is prompted to generate multiple potential solutions or ideas. If an idea is assessed as unviable, the model ceases further exploration along that path.

        3. **Solution Evaluation**: Upon reaching one or more solutions, the model evaluates them to select the most promising one. If necessary, it continues to the next step, refining its approach based on previous evaluations.

        Here is an example of a tree-of-thought prompting.
        """
    )
    return


@app.cell
def _(ollama, params_llm):
    _prompt = """ Imagine three different experts are answering this question. All experts will write down 1 step of their thiking, then share it with the group. Then, all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave. The question is

    {question}

    Make sure to dicuss the results.

    Format:

    Step <number> -----
    Expert A: <thinking>
    Expert B: <thinking>
    ...

    """

    _question = "Q: A kid has 21 stickers, gives away 20 stickers, and then gets 6 new stickers. How many stickers does the kid have now?"

    # Harder question
    #_question = "Mary drove 120 miles from her home to another city at an average speed of 40 mph and returned home at an average speed of 60 mph. What was her average speed for the entire round trip?"

    _prompt = _prompt.format(question = _question)

    _response = ollama.generate(prompt = _prompt, **params_llm)

    print(_response.response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DSPy
        We have learned how to guide language models to perform tasks by crafting specific prompts. But, we should not forget that what's more crucial is "what to do" rather than "how to do", which is the central idea of DSPy.

        [DSPy](https://dspy.ai/) is a Python framework that enables developers to define desired outcomes through declarative programming, allowing the framework to handle the underlying mechanics. This method abstracts away the intricacies of prompt engineering, facilitating the creation of modular and self-improving language model pipelines.

        ## Overview

        DSPy has three main components: Signature, Module, and Optimizer.

        1. **Signatures**: Declarative specifications that define the input and output behavior of a module.

        2. **Modules**: Reusable building blocks that abstract various prompting techniques, such as chain-of-thought reasoning or retrieval-augmented generation.

        3. **Optimizers (Teleprompters)**: Automated tools that refine prompts and parameters to enhance the performance of LLM pipelines.

        We will walk through each of these components in this example.

        ## Set up DSPy with Ollama
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Signature
        Signature defines the input and output of an LLM model.
        For example, the following signature defines a model that takes a text as input and outputs a sentiment.
        """
    )
    return


@app.cell
def _(dspy):
    qa_signature = dspy.Predict("text -> sentiment")
    _text = qa_signature(text = "I love DSPy!")
    print(_text)
    return (qa_signature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Under the hood, DSPy uses a prompt template to generate the prompt for the LLM for the given signature. You can inspect the prompt as follows:""")
    return


@app.cell
def _(lm):
    print(lm.inspect_history(n=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can create a more complex signature by defining subclassing the Signature class.""")
    return


@app.cell
def _(dspy):
    class QASignature(dspy.Signature):
        """Given a question, answer the question for the user. Take into account the context."""
        question = dspy.InputField(desc="The question")
        context = dspy.InputField(desc="The context for the question")
        answer = dspy.OutputField(desc="The answer to the question")

    qa_custom_signature = dspy.Predict(QASignature)
    _text = qa_custom_signature(question = "What is a famous landmark in Cambridge?", context = "We are traveling in the United States")
    print(_text)
    return QASignature, qa_custom_signature


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that the docstring of the signature is used to generate the prompt for the LLM.""")
    return


@app.cell
def _(lm):
    print(lm.inspect_history(n=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Reasoning is also supported in DSPy.""")
    return


@app.cell
def _(QASignature, dspy):
    _qa = dspy.ChainOfThought(QASignature)
    _text = _qa(question = "What is a famous landmark in Cambridge?", context = "We are traveling in the United States")
    print(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Module
        A **module** in DSPy is a reusable component defining a specific language-model task through structured inputs (signature), instructions, and output formatting (prefixes).
        For example, the following module defines a model that can answer questions based on a context.
        """
    )
    return


@app.cell
def _(dspy):
    class QAModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought("context, question -> answer")
            self.classifier = dspy.ChainOfThought("context, question, answer -> classification")

        def forward(self, question, context):
            answer = self.qa(context=context, question=question)
            classification = self.classifier(context=context, question=question, answer=answer)
            return classification

    _qa = QAModule()
    _text = _qa(question = "What is a famous landmark in Cambridge?", context = "We are traveling in the United States")
    print(_text)
    return (QAModule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Optimizer: Prompt Optimization
        DSPy provides prompt optimization for language models, which automates the process of finding the best prompt for a given task.

        /// warning | The optimization demo in a separate notebook.
        Since the optimization is computationally intensive, we will showcase it in a separate notebook at [dspy-prompt-optimization.py](https://static.marimo.app/static/dspy-prompt-optimization-52my).
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// tip | Framework for programming LLMs

        [TextGrad](https://textgrad.com/) is an alternative to DSPy. A central idea of TextGrad is backpropagation based on "natural language" generated by LLMs. Unlike DSPy, it offers less abstracted APIs, which allows a more fine-grained customization. Additionally, it can do **instance optimization**, an optimization on not the prompt but a solution from LLM. For example, you can provide your write up as few-shot examples and prompt the LLM to rewrite a given text in your writing style. Using TextGrad, we can create an iterative loop of rewriting and self-refletion by an LLM to create a rewriten text in your writing style.

        ![](https://anth.us/static/d604518c30b41a6907d7ac24aeec1cee/186a7/TextGrad.png)


        [Adaflow](https://github.com/SylphAI-Inc/AdalFlow) is a new framework that offers a more cleaner APIs than DSPy (in my personal opinion). It offers a different level of abstractions, offering a fine-grained customization as well as easy-to-use (but abstracted) functions to quickly mock an LLM application.

        ![](https://raw.githubusercontent.com/SylphAI-Inc/AdalFlow/main/docs/source/_static/images/adalflow-logo.png)

        ///
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
