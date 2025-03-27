# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch==2.6.0",
#     "transformers==4.49.0",
# ]
# ///

import marimo

__generated_with = "0.11.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    _fig = mo.md(
        "![](https://media.licdn.com/dms/image/v2/D4E22AQFZFRSwwzCSqQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1725003016027?e=2147483647&v=beta&t=oBH1s4V8N0wKCOJQakA_wrwgFrixs56S0s_QafZOvbA)"
    )
    _text = mo.md(
        """
        # Generative Pre-trained Transformer (GPT)
        <center>Sadamori Kojaku</center>
        """
    )
    mo.hstack([_text, _fig], widths=[1, 1], justify="center", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # What is GPT?

        ![GPT architecture](https://heidloff.net/assets/img/2023/02/transformers.png)

        - GPT (Generative Pre-trained Transformers) consists of a variant of the decoder-transformers

        - Many GPTs have been developed, with increasing capabilities, driven mostly by the increasing number of parameters.


        ![](https://preview.redd.it/p-converting-gpt-to-llama-step-by-step-code-guide-v0-qowi1sf12krd1.jpg?width=4286&format=pjpg&auto=webp&s=9ddaad2249f1b1b8084dab7f1279ba48927dda83)



        /// tip | Scaling law
        - Language model performance improves *predictably* as models get larger, following simple mathematical relationships (power laws).
        - The larger the model, the better it performs - and this improvement is reliable and measurable.
        - Larger models are more efficient learners - they need less training data and fewer training steps to achieve good performance.

        <center><img src="https://miro.medium.com/v2/resize:fit:1400/1*5fsJPwvFjS7fo8g8NwsxNA.png" width="500"></center>

        ///


        ## GPT ~ Autoregressive model

        ![](https://media.licdn.com/dms/image/v2/D4E22AQFZFRSwwzCSqQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1725003016027?e=2147483647&v=beta&t=oBH1s4V8N0wKCOJQakA_wrwgFrixs56S0s_QafZOvbA)

        GPT is an **autoregressive (causal) language modeling**

        Given a sequence of tokens $(x_1, x_2, ..., x_n)$, the model is trained to maximize the likelihood:

        $$
        P(x_1, ..., x_n) = \prod_{i=1}^n P(x_i|x_1, ..., x_{i-1})
        $$

        To generate text, the model predicts the next token given all previous tokens $(x_1, ..., x_{i-1})$. And then, the predicted token is appended to the sequence $(x_1, ..., x_{i-1}, x_{i})$, and the process is repeated until:

        1. the model predicts a special token called "end-of-text" token, e.g., <|endoftext|>
        2. the model generates a specific number of tokens, specified by the user.

        ## Sampling the most likely sequence

        We are interested in sampling the most likely sequence from the model.

        $$
        \arg\max_{x_1, ..., x_n} P(x_{\ell + 1}, \ldots, x_n \mid x_1, ..., x_{\ell})
        $$

        where:

        - $x_1, ..., x_{\ell}$ is the prefix of the sequence (the context).
        - $x_{\ell + 1}, \ldots, x_n$ is the sequence to be sampled (the generated text).

        However, this is not easy because there are many tokens that are statistically dependent on each other. **GPT provides us a (conditional) probability distribution of tokens but not the joint probability distribution. It is our job to sample a token sequence from the distribution.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Greedy sampling

        - **Greedy sampling** is a heuristic that always picks the highest probability token.
        - Deterministic but can lead to repetitive or trapped text.
        - For example, if the model predicts "the" with high probability, it will always predict "the" again.

        ## Let's try it out 🚀!

        We will use a small GPT model and generate text from it using greedy sampling. For this demonstration, we'll utilize the `gpt2` model from Hugging Face, accessed through the convenient `pipeline` interface provided by the `transformers` library.

        /// tip | What is pipeline?
        The `pipeline` function simplifies using pre-trained models by abstracting away model loading, tokenization, and post-processing.

        For example, without pipeline, we need to write the following code to generate text from the `gpt2` model:

        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        inputs = tokenizer("Hello, I'm a", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ```

        With pipeline, the code is much simpler (see the next cell 😉)
        ///
        """
    )
    return


@app.cell
def _():
    # Import the pipeline from transformers
    import torch
    from transformers import pipeline

    # Define the generator function
    generator = pipeline(
        "text-generation",  # the task
        model="gpt2",  # the model
        device="mps",
    )  # "mps", "cpu", or "cuda" depending on your hardware

    greedy_output = generator(
        "Hi there! ",
        do_sample=False,
        max_new_tokens=20,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    print(greedy_output[0]["generated_text"])
    return generator, greedy_output, pipeline, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        The output is often repetitive 🙃 because it always select the most probable token at each step.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Beam Search
            **Beam search** finds the best sequence by exploring multiple options at each step.

            1. The algorithm tracks $k$ best sequences (called *beams*) at a time.
    
            2. At each step, it predicts the top $B$ next tokens for each sequence.
    
            3. It keeps the $k$ most likely sequences and discards the rest.
    
            4. This continues until a stopping condition is met (e.g., max length or end token).

            ![](https://towardsdatascience.com/wp-content/uploads/2021/04/1tEjhWqUgjX37VnT7gJN-4g.png)

            Let's try it out!
        """
    )
    return


@app.cell(hide_code=True)
def _(generator):
    beam_output = generator(
        "Hi there! ",
        do_sample=False,
        max_new_tokens=20,
        pad_token_id=generator.tokenizer.eos_token_id,
        num_beams=10,  # number of beams
        num_return_sequences=10,
    )

    # Print the output
    for i in range(len(beam_output)):
        print(beam_output[i]["generated_text"])
    return beam_output, i


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Question**: Can we make a case in which beam search finds an optimal and suboptimal path (whose joint probability is less than the maximum joint probability)?"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## From Deterministic to Stochastic Sampling

        - **Deterministic Methods**
            - Greedy and beam search can get stuck in loops
            - Predictable but repetitive outputs

        - **Stochastic Sampling**
            - Introduces randomness to avoid loops 🎲
            - But can lead to incoherent text 👾

        - **Key Challenge**
            - Expensive computation of the probabilities of all tokens
            - *Controlling* randomness is crucial for quality output

        ### Two Popular Stochastic Sampling Methods

        To reduce the computational cost, we can pre-sample a few tokens, instead of all tokens, with the highest probability. The two popular methods are:

        - **Top-k Sampling** selects the k most likely tokens at each step.

        - **Nucleus Sampling** selects the smallest set of tokens with the highest cumulative probability exceeding a threshold $p$ (typically 0.9).


        /// note | Nucleus sampling
        **Nucleus sampling** dynamically adjusts by selecting fewer tokens when the distribution is concentrated and more tokens when the model is uncertain about predictions.

        ![](https://storage.googleapis.com/zenn-user-upload/8p2r9urhtn5nztdg6mnia3toibhl)
        ///

        To control the **randomness**, we can use **temperature** as follows.

        $$
        p_i = \\frac{\exp(z_i/\\tau)}{\sum_j \exp(z_j/\\tau)}
        $$

        where $z_i$ is the logit of the $i$-th token, and $\tau$ is the temperature.

        - The temperature $\\tau$ controls the concentration of the probability distribution.
        - Lower temperatures make the distribution more peaked, leading to more focused outputs.
        - Higher temperatures make the distribution more flat, producing more diverse but potentially less coherent text.

        ![](https://cdn.prod.website-files.com/618399cd49d125734c8dec95/6639e35ce91c16b3b9564b2f_mxaIPcROZcBFYta1I0nzWjlGTgs-LxzUOE3p6Kbvf9qPpZzBh5AAZG7ciRtgVquhLTtrM8ToJdNd-ubXvuz8tRfrqBwSozWHCj457pm378buxz2-XrMfWzfSv3b793QP61kLxRKT299WP1gbas_E118.png)

        Let's try it out!

        <center><b>Top-k Sampling</b></center>
        """
    )
    return


@app.cell
def _(generator):
    top_k_output = generator(
        "Hi there! ",
        do_sample=True,
        max_new_tokens=20,
        pad_token_id=generator.tokenizer.eos_token_id,
        top_k=10,  # set 1 to make it equivalent to greedy
    )
    print(top_k_output[0]["generated_text"])
    return (top_k_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""<center><b>Nucleus Sampling</b></center>""")
    return


@app.cell
def _(generator):
    top_p_output = generator(
        "Hi there! ",
        do_sample=True,
        max_new_tokens=20,
        pad_token_id=generator.tokenizer.eos_token_id,
        top_p=0.95,  # set 0 to make it equivalent to greedy
    )
    print(top_p_output[0]["generated_text"])
    return (top_p_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""<center><b>Temperature</b></center>""")
    return


@app.cell
def _(generator):
    for _tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
        output_tau = generator(
            "Hi there! ",
            do_sample=True,
            max_new_tokens=20,
            pad_token_id=generator.tokenizer.eos_token_id,
            temperature=_tau,  # set 0 to make it equivalent to greedy
        )
        print(f"tau = {_tau}: {output_tau[0]['generated_text']}")
    return (output_tau,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <center><b>Everything together</b></center>
              We can combine all the methods together to further limit the tokens to be sampled (e.g., they must be among the top k and top p tokens) with randomness control (e.g., temperature).
        """
    )
    return


@app.cell
def _(generator):
    for _tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
        _output = generator(
            "Hi there! ",
            do_sample=True,
            max_new_tokens=20,
            pad_token_id=generator.tokenizer.eos_token_id,
            temperature=_tau,  # set 0 to make it equivalent to greedy
            top_k=10,
            top_p=0.95,
        )
        print(f"tau = {_tau}: {_output[0]['generated_text']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Summary

              In this notebook, we've explored different text generation strategies for language models:

              1. **Greedy Sampling**: Always selects the most probable token at each step. Simple but often leads to repetitive text.

              2. **Beam Search**: Explores multiple possible sequences simultaneously, keeping track of the k most likely sequences.

              3. **Top-k Sampling**: Restricts sampling to only the k most likely tokens at each step, introducing controlled randomness.

              4. **Top-p (Nucleus) Sampling**: Dynamically selects from the smallest set of tokens whose cumulative probability exceeds threshold p.

              5. **Temperature Sampling**: Controls the randomness of predictions by scaling the logits before applying softmax. Lower temperature (τ < 1) makes the distribution more peaked, while higher temperature (τ > 1) makes it more uniform.

              These techniques can be combined to achieve the desired balance between diversity and quality in generated text. The optimal approach depends on your specific application and requirements.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(generator):
    _text = "Could you make a song?"

    _top_p_output = generator(
        _text,
        do_sample=True,
        max_new_tokens=20,
        pad_token_id=generator.tokenizer.eos_token_id,
        top_p=0.95,  # set 0 to make it equivalent to greedy
    )
    print(_top_p_output[0]["generated_text"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
