# Prompt Tuning Exercise


## Exercise 1: 🎲 Random Number Challenge! 🎲

Can you trick an LLM into being a random number generator? Let's find out!

Your mission:
- Create a clever prompt that makes an LLM generate at least 100 normally-distributed random numbers
- Numbers should be comma-separated (like: 0.5,-1.2,0.8,...)
- Your numbers need to pass a statistical test (KS-test with p-value > 0.20)
- Ask Gemma3 27B to create an SVG picture of a neural network
  - Try it on [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat) or [OpenRouter](https://openrouter.ai/google/gemma-3-27b-it:free) (both free!)
- No cheating! The LLM can't use tools like Code Interpreter or calculators
  - (If you cannot access, then other models are also fine 🙂. But the task might become a bit easier and not fun.)

**Tips for success:**
- Ask for just the numbers - no brackets, periods, or other characters
- You might want to ask the LLM to think about how normal distributions work before generating the numbers


<div>
<marimo-iframe data-height="600px" data-show-code="false">

```python
import marimo as mo
import altair as alt
import pandas as pd
import scipy.stats as stats
import numpy as np
```

```python
text_area = mo.ui.text_area(placeholder = "Enter numbers separated by commas", value = "1,2,3,4,5,6,7,8,9,10")
button = mo.ui.button("Runt test")

mo.vstack([text_area, button])
```

```python
try:
    numbers = np.array([float(num.strip()) for num in text_area.value.split(",")])
    if len(numbers) >= 100:
        # KS test
        pval = stats.kstest(numbers, stats.norm(loc=0.0, scale=1.0).cdf)[1]
        test_result = "The numbers are normal distributed (p-value = {:.2f})".format(pval) if pval > 0.20 else "The numbers are not normal distributed (p-value = {:.2f})".format(pval)
        message = mo.callout(test_result, kind = "success" if pval > 0.20 else "danger")
    else:
        message = mo.callout("The number of samples is too small. Need at least 100 samples.", kind = "warn")

    # Convert the numbers to a DataFrame for Altair
    df = pd.DataFrame({'value': numbers})

    # Create an Altair histogram
    fig = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', bin=alt.Bin(maxbins=30)),
        y='count()'
    ).properties(
        title='Histogram of Values'
    )

    mo.hstack([fig, message])
except:
    message = mo.callout("Parse failed. Please check if your input follows the specified format.", kind = "danger")
    fig = None

mo.hstack([fig, message]) if fig is not None else message
```

</marimo-iframe>
</div>

## Exercise 2: 🎨 Make an LLM Draw a Neural Network! 🎨

Your fun mission:
- Ask Gemma3 27B to create an SVG picture of a neural network
  - Try it on [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat) or [OpenRouter](https://openrouter.ai/google/gemma-3-27b-it:free) (both free!)
- Copy the SVG code and paste it into [SVG Viewer](https://www.svgviewer.dev/)

Make sure your neural network has:
- Same-colored neurons in each layer
- Connections between all neurons
- Labels for "Input layer", "Hidden layer", and "Output layer"

Ready, set, prompt! 🚀

<div style="text-align: center;">
  <img src="https://www.researchgate.net/publication/329777725/figure/fig2/AS:705569090465794@1545232188535/A-simple-neural-network-diagram-with-one-hidden-layer.ppm" width="50%">
</div>

<script src="https://cdn.jsdelivr.net/npm/@marimo-team/marimo-snippets@1"></script>