# **Exploring Bias Logit Warping** 🎯🚀

Ever wondered how to make a language model dance to your tune? Welcome aboard the spaceship of Bias Logit Warping, your magic wand for gaining razor-sharp control over a language model's responses. It's a game-changer that lets you tinker with output probabilities for specific words or phrases, creating a tailor-made text-generation experience.

<p align="center">
  <img src="https://cdn.discordapp.com/attachments/847959427424452639/1120352071128977408/Screen_Recording_2023-06-17_at_23.57.01.gif" alt="Softmax applied to logits" width="600px" />
</p>

It's like adding a thumb on the scale to tip the balance in favour of certain words.

## The Bias Effect: Meet Greg 🌟

Imagine we have a prompt that says `"Hello, my name is "` as a prompt. With a little bias magic, we can influence the story:

- Without bias: `{}` <br>
    Output: `"iam samuel sailor"`
- Bias towards `{"Greg": +6}` <br>
    Output: `"Greg Gregg"`
- Higher bias towards `{"Greg": +13.37}` <br>
    Output: `"Greg Greg Greg Greg Greg Greg Greg Greg..."`

<p align="center">
  <img src="https://media.tenor.com/xFtLd_pUrkcAAAAC/succession-greg.gif" alt=Greg" width="600px" />
</p>

## Unraveling Logits and Bias 📊🎩

### Language Models, Logits, and the Magic of Softmax 🧠🔮

Language Models (LMs) are like mysterious black boxes spinning out human-like text. At the heart of this wizardry is the delicate pas de deux between 'logits' and the 'softmax' activation function.

Logits are the raw, unprocessed votes that the model gives for each potential next word in a sequence. However, these votes need a little makeover, a transformation into probabilities.

Enter the 'softmax' function, our logit beautician! It turns these plain Jane logits into probabilities, all ready for the grand text-generation ball. But, the word with the highest probability doesn't always get to be the queen of the ball. To avoid an echo chamber, the model uses sampling techniques for a more varied, creative, and engaging text.

### Bias Logit Warping: Shaping the Narrative ⚖️🛠

How do we pull the strings of this narrative? Enter 'Bias Logit Warping' - our handy tool to tweak logits of specific words, enhancing or reducing their chances of being selected. It's like having a personal remote control for your text output!

This technique modifies the logits of specific tokens by infusing a 'bias' value into them. Depending on the bias value, it can enhance (positive bias) or diminish (negative bias) the probability of a particular w being selected. It's akin to tipping the scale to favour/guarantee/ban certain tokens!

## FLAN and T5 Models 🤖🔎

To understand Bias Logit Warping, we need to take a closer look at FLAN and T5 models.

FLAN (Finetuned LAnguage Net), Google's brainchild, is a Transformer model that thrives on zero-shot learning tasks. It's like a sponge soaking up knowledge from a limited set of examples.

Meanwhile, T5 (Text-to-Text Transfer Transformer) is a chameleon, transforming all NLP problems into a text-to-text format. It's a one-stop-shop for tasks like translation, summarization, and question-answering.

## The ABCs of Word-Level Tokenization 📝💡

Tokenization is like a text's chopping board, dividing it into digestible pieces called tokens. FLAN T5, with its trusty word-level tokenization, splits text into words or subwords.

Let's take a close look at `"unhappiness"`. In the world of FLAN T5, this word gets artfully dissected into something akin to `["un", "<dash>", "happy", "iness"]`. Such fine-grained tokenization not only helps the model unravel the intricacies of complex words but also plays a pivotal role when we're ready to season our logits with bias. Conveniently, the way this model tokenizes often aligns the tokens with the words we humans naturally want to influence, as illustrated by our `"Greg"` example!

## Pulling the Bias Levers: Word-Level Biasing and Logit Warping 🧪🎮

Think of word-level biasing as boosting a word's charisma, influencing its chances of showing up in the text. When we sprinkle bias onto a word's logit, we're dialling up its charm factor in the model's eyes.

FLAN T5, being a part of the Transformer family, considers all words when making predictions. Add some bias to the mix, and you'll notice a ripple effect. Not only does it boost a word's chances, but it also rearranges the words' relationships, leading to a fresh new output!

## The Beauty of Math: Softmax and Bias Unveiled 🧮💫

Here's a quick tour of the math driving the magic. The softmax function converts logits into probabilities:

![Softmax Function](https://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28x%29_i%20%3D%20%5Cfrac%7Be%5E%7Bx_i%7D%7D%7B%5Csum_j%20e%5E%7Bx_j%7D%7D)

In Bias Logit Warping, we sprinkle some bias into the logits:

![Biased Logit](https://latex.codecogs.com/gif.latex?%5Ctext%7Blogit%7D_%7B%5Ctext%7Bbiased%7D%7D%20%3D%20%5Ctext%7Blogit%7D%20%2B%20%5Ctext%7Bbias%7D)

And voila! We get the biased logits feeding into the softmax function:

![Softmax with Bias](https://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28x%20%2B%20b%29_i%20%3D%20%5Cfrac%7Be%5E%7Bx_i%20%2B%20b_i%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BV%7D%20e%5E%7Bx_j%20%2B%20b_j%7D%7D)

In this math magic, `x` is the logit, `b` is the bias, and `V` is the vocabulary size. Play around with `b_i` to increase or decrease the softmax probability for any token (`i`).

## One Last Thought 🎓🌈

Even though biasing logits is like having a language model steering wheel, remember that the journey is full of surprises. The model's randomness ensures an exciting, unpredictable ride every time you spin the bias dial. So buckle up and enjoy your Bias Logit Warping adventure!

___

<br>

# Appendices

We're basing our work on [OpenAI's API](https://platform.openai.com/docs/api-reference/chat/create#chat/create-logit_bias) which introduces `logit_bias`. This nifty tool lets us tweak the likelihood of certain tokens popping up in the model's completion.

Here's a quick tour: `logit_bias` accepts a JSON object that maps tokens (identified by their token ID in the tokenizer) to a bias value that could range anywhere between -100 to 100. Mathematically speaking, this bias is added to the logits that the model produces before it starts sampling. The impact varies depending on the model, but generally, values between -1 and 1 mildly increase or decrease a token's chances of being chosen. On the other hand, extreme values like -100 or 100 pretty much blacklist or guarantee the token's selection.

## Getting Hands-on with Cog Locally 🔧🏠

Before we dive in, ensure you have [Cog](https://github.com/replicate/cog) ready to roll.

Start by cloning the repo like a champ:
```
git clone git@github.com:zsxkib/replicate-lil-flan.git
```

Then, fire up your terminal and let the building commence:
```zsh
cog build
```
Voila! You're now officially building with Cog.

## Making a Prediction Happen 🏃‍♂️🎯

It's prediction time! Once you've nailed the Cog build, you can set the prediction ball rolling with this command:
```
cog predict -i prompt="Hello, my name is " -i bias_dict_str="{'Greg': +5.5}"
```
Ready to meet Greg? Let's go:
```zsh
Running prediction...
Greg Gregg
```
There you have it - making predictions is as easy as pie! 

##### **Special thanks *[@daanelson](https://github.com/daanelson)* for coming up with this!**
