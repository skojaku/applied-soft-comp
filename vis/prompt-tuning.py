import marimo

__generated_with = "0.11.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Prompt Tuning for Language Models from Basics to Mastery
        <center>Sadamori Kojaku</center>

        Getting desired responses from LLMs requires carefully designed prompts. Many prompting techniques have been developed, and here we will learn non-exclusive but basic ingredients of a prompt.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Set up

        ![](https://miro.medium.com/v2/resize:fit:1168/0*-9J6J-vqMDaHr174.png)


        We will use [ollama](https://ollama.com/) to run **a small language model** (SLM). Ollama is a lightweight runtime engine for running large language models locally. With ollama, you can chat LLMs running on your machine through many programming languages, including Python.

        /// note | Why open source small language models?

        Open source small language models help democratize AI by making advanced, language processing widely accessible.

        Some non-exclusive lists of open source SLMs are:

        1. **[Gemma (Google DeepMind)](https://ai.google.dev/gemma)** – Lightweight and optimized for efficiency.
        2. **[DeepSeek R1 (deepseek)](https://github.com/deepseek-ai)** - Open-source model series optimized for reasoning, efficiency, and multilingual support.
        3. **[Phi series (Microsoft)](https://azure.microsoft.com/en-us/products/phi)** – Compact model designed for reasoning tasks with minimal compute.
        4. **[Mistral 7B](https://mistral.ai/news/announcing-mistral-7b)** – High-performance dense transformer with strong generalization.
        5. **[Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts)** – Mixture-of-experts model, using only a subset of parameters per inference.
        6. **[TinyLlama 1.1B](https://github.com/jzhang38/TinyLlama)** – Ultra-small LLaMA variant for resource-constrained environments.
        7. **[StableLM (Stability AI)](https://github.com/Stability-AI/StableLM)** – Open-source models for conversational AI and code generation.

        In this notebook, we will use gemma3 but you are free to choose any SLM you like 😉

        ///
        """
    )
    return


@app.cell
def _():
    # from langchain_ollama import ChatOllama
    import ollama

    # Get a response from the model
    params_llm = {"model": "gemma3", "options": {"temperature": 0.3}}

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
        r"""
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
        r"""
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


app._unparsable_cell(
    r"""
    <center><b>Few shot</b></center>
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


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


app._unparsable_cell(
    r"""
    <center><b> Without chain prompting</b></center>
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


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


@app.cell(hide_code=True)
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
        # Automatic Prompt Optimization

        Crafting effective prompts remains challenging; even subtle wording changes can significantly influence language model outputs. Historically, prompt design has relied on manual, trial-and-error adjustments—analogous to early neural network training before backpropagation automated parameter tuning.

        Automatic prompt optimization simplifies the prompt-tuning process. Frameworks such as [DSPy](https://github.com/stanfordnlp/dspy) and [AutoPrompt](https://github.com/Eladlev/AutoPrompt) facilitate this automation. Here, we focus on [TextGrad](https://textgrad.com/), a framework leveraging textual gradients—a natural-language counterpart to numerical gradients used in neural network backpropagation.

        ## TextGrad
        ![](https://anth.us/static/d604518c30b41a6907d7ac24aeec1cee/186a7/TextGrad.png)
        ![](https://textgrad.readthedocs.io/en/latest/_images/analogy.png)

        The core idea of TextGrad is an analogy with backpropagation:

        1. Prompts targeted for optimization are called `differential`  text variables  (metaphorically, not mathematically).

        2. A network of prompts (e.g., chained prompts) is called a `model`

        3. Textual input data are fed into the model, producing textual outputs (Forward pass).

        4. Outputs are evaluated, generating natural-language feedback (Backward pass).

        4. Explicit textual guidance provided by an LLM, informing prompt adjustments (Gradient).

        ## Results

        ![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9af6abbb-0a9a-4056-b3a3-da0b9ae344f7_887x1204.png)

        TextGrad has demonstrated to be consistently very effective across various applications. Here are some remarkable results:

        - **Coding:** By refining solutions to complex coding challenges, TextGrad achieved a 20% relative improvement in performance on LeetCode Hard problems, surpassing existing methods.

        - **Problem Solving:** In the Google-Proof Question Answering benchmark, TextGrad enhanced GPT-4o's zero-shot accuracy from 51% to 55% by iteratively refining responses during testing.

        - **Reasoning:** Through prompt optimization, TextGrad elevated GPT-3.5's performance to levels comparable with GPT-4 in several reasoning tasks, highlighting its efficacy in enhancing logical processing.

        - **Chemistry:** The framework successfully designed new small molecules exhibiting desirable drug-like properties and favorable in silico binding affinities, indicating potential in drug discovery.

        - **Medicine:** TextGrad optimized radiation treatment plans for prostate cancer patients, achieving targeted dosages while minimizing side effects, thereby improving patient outcomes.


        ## Example with Ollama

        To illustrate TextGrad, consider our previous task: generating a sales pitch for Steve. The first step is to create so-called an `engine`, an interface that connects a language model and TextGrad.
        """
    )
    return


@app.cell
def _():
    from openai import OpenAI

    # Setup
    client = OpenAI(
        base_url="http://0.0.0.0:11434/v1",
        api_key="ollama",
        timeout=300.0,
    )
    return OpenAI, client


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Let's define our `objective function` in natural language. This defines the task the prompts will be optimized for.""")
    return


@app.cell
def _():
    eval_prompt = """Evaluate how convincingly this response reflects Steve Jobs' distinctive style—visionary, persuasive, and succinct—while effectively tailoring his pitch to the context of interviewing at a phone repair shop. Provide a score from 1 (poor) to 5 (excellent) and briefly justify your rating."""

    initial_prompt = "You are Steve Jobs preparing for a job interview at a phone repair shop. Generate a sales pitch for your job interview."
    return eval_prompt, initial_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's define the `model`, a network of prompts and LLMs. In this example, we focus on the most simple model, one prompt and one LLM.""")
    return


@app.cell
def _(client, initial_prompt):
    from textgrad.engine.local_model_openai_api import (
        ChatExternalClient,
    )
    import textgrad as tg

    class RolePlayLLM:
        def __init__(
            self,
            client,
            model_name,
            feedback_model_name,
            initial_prompt,
            max_tokens = 1024 
        ):

            self.engine = ChatExternalClient(
                client=client, model_string=model_name, max_tokens=max_tokens
            )
            self.feedback_engine = ChatExternalClient(
                client=client, model_string=feedback_model_name, max_tokens = max_tokens
            )
            tg.set_backward_engine(
                self.feedback_engine, override=True
            )
            self.user_prompt = tg.Variable(
                initial_prompt,
                requires_grad=True,
                role_description="prompt to guide the LLM for interview pitch",
            )
            self.model = tg.BlackboxLLM(
                engine=self.engine, system_prompt=None
            )

        def forward(self, **kwargs):
            return self.model(self.user_prompt)

        def parameters(self):
            return [self.user_prompt]


    model = RolePlayLLM(
        client,
        model_name="gemma3:27b",
        feedback_model_name="gemma3:27b",
        initial_prompt=initial_prompt,
    )
    return ChatExternalClient, RolePlayLLM, model, tg


app._unparsable_cell(
    r"""
    Then, set the loss function and the optimizer, just like PyTorch 😉!
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(eval_prompt, model, tg):
    # Loss function
    loss_fn = tg.TextLoss(eval_prompt)

    # Set optimizer
    optimizer = tg.TGD(parameters=model.parameters())

    # Optimization loop
    num_iterations = 3

    prompt_list = []
    result_list = []

    for i in range(num_iterations):
        # Forward pass
        prediction = model.forward()

        loss = loss_fn(prediction)

        result_list.append(prediction.value)
        prompt_list.append(model.user_prompt.value)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print the updated prompt
        print(f"Updated prompt:\n{model.user_prompt.value}\n")
    return (
        i,
        loss,
        loss_fn,
        num_iterations,
        optimizer,
        prediction,
        prompt_list,
        result_list,
    )


@app.cell
def _(mo, prompt_list, result_list):
    mo.accordion({"Iteration %d" % i : "<b>Prompt</b>:{prompt}<br><br><b>Result</b>:{result}".format(prompt = prompt_list[i], result = result_list[i]) for i in range(len(prompt_list))})
    return


@app.cell
def _(mo, prompt_list, result_list):
    import difflib
    import re

    def create_simple_markdown_diff(prompt_list, result_list):
        """
        Create a simple markdown diff between prompts and display in Marimo accordion.
        - Added sentences are in bold
        - Deleted sentences are struck through
        """
        accordion_items = {}

        def split_into_sentences(text):
            """Split text into sentences using regex"""
            text = re.sub(r'([.!?])\s+', r'\1\n', text)
            text = re.sub(r'([.!?])"', r'\1"\n', text)
            return [s.strip() for s in text.split('\n') if s.strip()]

        for i in range(len(prompt_list)):
            markdown_content = f"### Prompt {i+1}\n\n"

            # For iterations after the first, show the diff
            if i > 0:
                prev_sentences = split_into_sentences(prompt_list[i-1])
                curr_sentences = split_into_sentences(prompt_list[i])

                # Find differences using difflib
                matcher = difflib.SequenceMatcher(None, prev_sentences, curr_sentences)

                # Process the differences
                diff_items = []
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        # Unchanged sentences
                        for sentence in prev_sentences[i1:i2]:
                            diff_items.append(sentence)
                    elif tag == 'delete':
                        # Deleted sentences
                        for sentence in prev_sentences[i1:i2]:
                            diff_items.append(f"~~{sentence}~~")
                    elif tag == 'insert':
                        # Added sentences
                        for sentence in curr_sentences[j1:j2]:
                            diff_items.append(f"**{sentence}**")
                    elif tag == 'replace':
                        # Replaced sentences (show as delete + insert)
                        for sentence in prev_sentences[i1:i2]:
                            diff_items.append(f"~~{sentence}~~")
                        for sentence in curr_sentences[j1:j2]:
                            diff_items.append(f"**{sentence}**")

                # Join the diff items
                markdown_content += "Changes from previous prompt:\n\n"
                markdown_content += "\n\n".join(diff_items)
            else:
                # First iteration - just show the text
                markdown_content += prompt_list[i]

            # Add the result
            markdown_content += f"\n\n### Result {i+1}\n\n"
            markdown_content += result_list[i]

            # Add to accordion items
            accordion_items[f"Iteration {i+1}"] = markdown_content

        return mo.accordion(accordion_items)

    # Example usage
    create_simple_markdown_diff(prompt_list, result_list)
    return create_simple_markdown_diff, difflib, re


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
