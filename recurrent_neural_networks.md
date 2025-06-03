# Recurrent Neural Networks (RNNs): Processing Sequential Data

## 1. What are Recurrent Neural Networks (RNNs)? (PYQ 5b - May 2024, PYQ 3b - May 2023, PYQ 5a - CBGS)

**Recurrent Neural Networks (RNNs)** are a class of artificial neural networks specifically designed to process **sequential data** or **time-series data**. Unlike traditional feedforward neural networks (like MLPs or CNNs) where inputs are assumed to be independent of each other, RNNs are built to recognize patterns in sequences of data where the order of elements matters.

**The Key Idea: "Memory" or Feedback Loop**

The defining characteristic of an RNN is its **internal memory** (often called the **hidden state**), which allows it to persist information from previous inputs in the sequence to influence the processing of current and future inputs. This is achieved through a **feedback loop** where the output from a previous step is fed back into the network as an input for the current step.

**Think of it as:** Reading a sentence. To understand the meaning of the current word, you need to remember the words that came before it. An RNN tries to mimic this by maintaining a memory of past information as it processes a sequence.

**Why are RNNs needed for sequential data?**
Feedforward networks process each input independently. They have no inherent memory of past inputs. This makes them unsuitable for tasks where context from previous elements is crucial:
*   **Language Modeling:** Predicting the next word in a sentence requires knowing the preceding words.
*   **Speech Recognition:** Interpreting a phoneme depends on the phonemes that came before it.
*   **Time Series Analysis:** Predicting future stock prices depends on past price trends.

RNNs solve this by introducing recurrent connections that allow information to loop back into the network, creating a sense of memory.

## 2. Architecture of a Simple RNN

A simple RNN (often called a "vanilla" RNN) has a relatively straightforward structure when viewed at a single time step, but its power comes from how this structure is repeatedly applied across a sequence.

**At a single time step `t`:**
*   **Input (`x_t`):** The element of the sequence at the current time step `t`.
*   **Hidden State (`h_t`):** The network's internal memory at time step `t`. It's calculated based on the current input `x_t` and the hidden state from the previous time step `h_{t-1}`.
*   **Output (`y_t`):** The prediction or output of the network at time step `t`. This can be optional at each time step, or only produced at the end of the sequence, depending on the task.

**The Recurrent Formula:**
The core of an RNN's operation can be described by these formulas:
1.  **Hidden State Calculation:**
    `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    *   `h_{t-1}`: Hidden state from the previous time step (t-1).
    *   `x_t`: Input at the current time step (t).
    *   `W_hh`: Weight matrix for the recurrent connection (hidden state to hidden state).
    *   `W_xh`: Weight matrix for the input connection (input to hidden state).
    *   `b_h`: Bias term for the hidden state calculation.
    *   `tanh`: Hyperbolic tangent activation function (a common choice, squashes values to between -1 and 1).

2.  **Output Calculation (Optional at each step):**
    `y_t = W_hy * h_t + b_y`
    *   `h_t`: Current hidden state.
    *   `W_hy`: Weight matrix for the output connection (hidden state to output).
    *   `b_y`: Bias term for the output calculation.
    (An activation function like Softmax might be applied to `y_t` for classification tasks).

**The Loop:**
Visually, an RNN cell can be depicted with a loop, indicating that the hidden state is passed from one time step to the next:

```
      x_t (Input)
        |
        V
    +-------+
----| RNN   |------ h_t (Current Hidden State) ----> y_t (Output)
|   | Cell  |   |
|   +-------+   |
|      ^        |
|______|________|
  h_{t-1} (Previous Hidden State)
```

**Unrolling the RNN in Time:**
To better understand how an RNN processes a sequence, it's helpful to "unroll" or "unfold" it in time. This means creating a separate copy of the RNN cell for each element in the input sequence. The hidden state is passed from one cell to the next.

For a sequence of length `T` (e.g., a sentence with `T` words `x_1, x_2, ..., x_T`):

```
Input:    x_1        x_2        x_3        ...      x_T
           |          |          |                   |
           V          V          V                   V
h_0 -> [RNN Cell] -> [RNN Cell] -> [RNN Cell] -> ... -> [RNN Cell] -> h_T
           |          |          |                   |
           V          V          V                   V
Output:   y_1        y_2        y_3        ...      y_T
```
*   `h_0` is the initial hidden state (often initialized to zeros).
*   Each `[RNN Cell]` is the same network cell, with the same set of weights (`W_hh`, `W_xh`, `W_hy`) applied at each time step.

## 3. How RNNs Process Sequences

1.  **Initialization:** The hidden state `h_0` is initialized (e.g., to zeros).
2.  **Step-by-Step Processing:** For each time step `t` from 1 to `T` (length of the sequence):
    a.  The RNN takes the current input `x_t` and the previous hidden state `h_{t-1}`.
    b.  It calculates the new hidden state `h_t` using the recurrent formula. This `h_t` captures information from all previous inputs `x_1, ..., x_t`.
    c.  It (optionally) calculates an output `y_t` based on `h_t`.
3.  **Final Output:** Depending on the task, the output might be the sequence of all `y_t` values, or just the final `y_T`, or the final hidden state `h_T` might be used for further processing.

**Shared Weights (Parameter Sharing):**
A crucial aspect of RNNs is that the same weight matrices (`W_hh`, `W_xh`, `W_hy`) and bias terms (`b_h`, `b_y`) are used at every time step. This has several advantages:
*   **Reduces Parameters:** The number of parameters doesn't grow with the length of the sequence.
*   **Generalization:** Allows the model to generalize patterns across different positions in the sequence. It learns a rule that can be applied at any point in time.

## 4. Applications of RNNs

RNNs excel in tasks involving sequential data:

*   **Natural Language Processing (NLP):**
    *   **Language Modeling:** Predicting the next word in a sentence (e.g., text generation in chatbots, autocomplete).
    *   **Machine Translation:** Translating a sentence from one language to another (encoder-decoder RNNs).
    *   **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) of a piece of text.
    *   **Text Summarization:** Generating a concise summary of a longer document.
*   **Speech Recognition:** Converting spoken audio into text. The audio is a sequence of sound waves over time.
*   **Time Series Prediction:**
    *   **Stock Market Prediction:** Forecasting future stock prices based on historical data.
    *   **Weather Forecasting:** Predicting future weather conditions.
*   **Video Analysis:** Understanding the content of videos by processing sequences of frames.
*   **Music Generation:** Composing new musical pieces.

**Example: Sentiment Analysis**
Input: A sentence (sequence of words) like "This movie was fantastic!"
1.  Each word is converted into a vector (word embedding).
2.  The RNN processes the sequence of word vectors one by one.
3.  The hidden state at each step tries to capture the meaning of the sentence so far.
4.  The final hidden state (or an output at the last step) is fed into a classifier (e.g., a fully connected layer with a softmax/sigmoid) to predict the sentiment (e.g., "positive").

## 5. Challenges with Simple RNNs

While powerful, simple RNNs face some significant challenges, especially when dealing with long sequences:

*   **Vanishing Gradient Problem:** During backpropagation through time (the training algorithm for RNNs), gradients can become extremely small as they are propagated back through many time steps. This means that the influence of earlier inputs on the current output diminishes, making it very difficult for the RNN to learn **long-range dependencies** (i.e., relationships between elements far apart in the sequence).
    *   **Example:** In the sentence "The cat, which already ate a lot of fish earlier in the day, was full," to predict "was full," the model needs to remember "cat" from many words ago. Vanishing gradients make this hard.

*   **Exploding Gradient Problem:** Conversely, gradients can also become extremely large, leading to unstable updates and divergence during training. This is often easier to handle (e.g., by gradient clipping).

These challenges, particularly the vanishing gradient problem, limit the ability of simple RNNs to effectively capture long-term memory. This led to the development of more sophisticated RNN architectures like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units), which are designed to better handle these issues.

## 6. Summary for Exams

*   **RNNs are for sequential data:** Text, speech, time series where order matters.
*   **Key Idea: Memory/Feedback Loop:** Achieved via a **hidden state (`h_t`)** that carries information from previous time steps (`h_{t-1}`) along with current input (`x_t`) to influence the current step.
*   **Architecture:** Recurrent connections; unrolled in time for visualization and training. Uses shared weights across time steps.
*   **Processing:** Takes sequence elements one by one, updating hidden state at each step.
*   **Applications:** NLP (language modeling, translation), speech recognition, time series.
*   **Challenge:** **Vanishing gradients** make it hard to learn long-range dependencies in simple RNNs.

RNNs provide a foundational way to introduce memory into neural networks, making them capable of understanding and generating sequential patterns. 