Exactly. This is the main piece we'd be customizing.

**Vanilla StyleTTS2 does not inherently know that `[sarcastic=0.8]` means "change the speaking style."** The bracketed text is just text unless we deliberately create a mechanism that turns those tags into style conditioning.

The good news is: **we do not need to redesign StyleTTS2 from scratch.** We can add a Kion-specific style-conditioning path around its existing architecture.

## 1. What vanilla StyleTTS2 does

At a simplified level, StyleTTS2 takes:

```text
text
  ↓
text/phoneme representation
  ↓
duration/alignment + acoustic modeling
  ↓
style/prosody information
  ↓
decoder
  ↓
audio
```

A major part of StyleTTS2 is its learned latent style representation. The original model can infer style from speech/reference information and uses a diffusion-based style mechanism to model variation in speech.

But your Kion system wants something more explicit:

```text
[sarcastic=0.8] Oh, wonderful.
```

The model needs to learn:

```text
sarcastic → style
0.8       → strength
```

rather than treating `"sarcastic"` as ordinary text.

---

# 2. We'd split the input into TEXT and STYLE

Suppose Kion's LLM outputs:

```text
[sarcastic-deadpan=0.7] Oh, fantastic. [concerned-mild=0.5] Wait, seriously?
```

Our preprocessing layer would turn that into something like:

```text
TEXT:

"Oh, fantastic. Wait, seriously?"
```

and:

```text
STYLE:

segment 1:
sarcastic-deadpan = 0.7

segment 2:
concerned-mild = 0.5
```

So instead of sending the literal tag to the phoneme encoder:

```text
❌ [ s a r c a s t i c ... ]
```

we do:

```text
                    Kion input
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Text parser          Style parser
             │                   │
             ▼                   ▼
      phonemes/tokens       style controls
             │                   │
             └─────────┬─────────┘
                       ▼
                  StyleTTS2
```

That's the first customization.

---

# 3. What exactly is a "style control"?

We need to create a small neural module that converts your tags into vectors.

For example:

```text
sarcastic-deadpan
        ↓
   embedding vector
        ↓
[ 0.13, -0.42, 0.71, ... ]
```

Then intensity scales that representation:

```text
style = embedding("sarcastic-deadpan") × 0.7
```

So:

```text
[sarcastic-deadpan=0.3]
```

and:

```text
[sarcastic-deadpan=0.9]
```

have the **same underlying style identity** but different strength.

Conceptually:

```text
             sarcastic-deadpan
                     │
                     ▼
              learned embedding
                     │
               × intensity
                     │
                     ▼
             style conditioning
```

---

# 4. Blends become very easy conceptually

Your input:

```text
[playful=0.7,teasing=0.5]
You really thought I'd let you do that?
```

gets parsed as:

```text
playful embedding × 0.7
              +
teasing embedding × 0.5
              ↓
        combined vector
```

Mathematically, something like:

$$
s = 0.7E_{\text{playful}} + 0.5E_{\text{teasing}}
$$

where `E` is a learned embedding for that style.

Then the model receives:

```text
text representation
        +
combined style representation
```

and generates the speech.

This is why I kept saying I don't want Kion's 37 styles to be treated as **37 mutually exclusive classes**.

The model can learn a continuous style space.

---

# 5. But your transitions require something more interesting

Suppose:

```text
[happy-strong=0.9] I finally fixed it!
[concerned-mild=0.6] Wait, why is the server still offline?
```

We don't have one style vector for the entire sentence.

We have:

```text
text token/phoneme positions:

I finally fixed it! | Wait, why is the server still offline?
          │                           │
          ▼                           ▼
      happy style              concerned style
```

So we'd create a **style sequence** aligned with the text.

Conceptually:

```text
Text:

I  finally  fixed  it  Wait  why  is  the  server  still  offline
│      │      │     │    │    │    │    │      │      │      │
└──────┴──────┴─────┘    └────┴────┴────┴──────┴──────┴──────┘
      HAPPY 0.9               CONCERNED 0.6
```

The style conditioning becomes something like:

```text
style_embedding_sequence:

[h h h h c c c c c c c]
```

where `h` and `c` are the corresponding style representations.

That gives the acoustic model information about **where the style changes**.

---

# 6. This is very important for your chunked streaming system

Your final runtime might receive:

```text
[sarcastic-deadpan=0.7] Yeah, that's definitely what I wanted.
```

then:

```text
[concerned-mild=0.6] But seriously, are you okay?
```

Since they're separate chunks, the parser simply creates a new style conditioning vector for the next synthesis call.

So we don't need the model to understand:

> "Why did Kion become concerned?"

The LLM already decided that.

The TTS simply receives:

```text
style = concerned-mild
strength = 0.6
```

and renders it.

---

# 7. Where does the actual "style" information come from during training?

This is the really important part.

You have:

```text
audio.wav
+
[sarcastic-deadpan=0.7] text
```

The audio itself contains:

* pitch
* rhythm
* duration
* energy
* emphasis
* pauses
* voice quality
* articulation
* other prosodic characteristics

StyleTTS2's existing architecture is already designed to learn speech style/prosody representations from audio. We don't have to manually label every pitch contour.

Our custom conditioning tells the model:

> "This acoustic style corresponds to `sarcastic-deadpan=0.7`."

So during training:

```text
                     training sample

[sarcastic=0.7] text ──────┐
                           ▼
                    Kion style encoder
                           │
                           ▼
                      StyleTTS2
                           ▲
                           │
                   target Kion audio
```

The model adjusts the style embedding/conditioning path so that the requested style produces acoustic characteristics matching the training recordings.

---

# 8. We'd probably use a learnable style embedding table

Something like:

```text
Style embeddings:

affectionate        → E1
authoritative       → E2
calm                → E3
concerned-mild      → E4
...
teasing             → E34
```

Then the intensity is applied separately.

For example:

```text
E_sarcastic × 0.4
E_sarcastic × 0.7
E_sarcastic × 1.0
```

The model therefore sees that these aren't three different emotions.

They're:

```text
same style
different magnitude
```

That is exactly what you want.

---

# 9. Your 37 styles don't necessarily need 37 completely independent vectors

This gets even more interesting.

Some of your styles are related:

```text
sarcastic-deadpan
sarcastic-biting
sarcastic-playful
dry-sarcasm
```

Rather than forcing them to be isolated, the network can learn relationships between them.

Conceptually:

```text
              SARCASTIC REGION
               /      |       \
          deadpan   dry     biting
                 \
               playful
```

Similarly:

```text
sad-mild → sad-strong → heartbroken
```

and:

```text
concerned-mild → concerned-strong
```

The embeddings can naturally occupy nearby regions.

---

# 10. But there is a problem with your current labels

You have:

```text
suprised-mild
suprised-strong
```

with `"suprised"` misspelled.

Because you're deliberately defining the syntax, **we should keep that spelling exactly throughout the dataset if that's what the current contract is**.

Later, if you want to rename it to `surprised`, that's a dataset/tokenizer migration rather than something we casually change halfway through.

Consistency matters more than spelling perfection here.

---

# 11. What happens during training with a transition?

Your synthetic audio might be:

```text
Audio:
happy Kion → concerned Kion
```

and metadata:

```text
[happy-strong=0.9] I finally fixed it!
[concerned-mild=0.6] Wait, why is it still broken?
```

Our parser creates:

```text
Text tokens:
I finally fixed it Wait why is it still broken

Style labels:
HHHHHHHHHCCCCCCCCCCCCCCCC
```

Then the model learns that the acoustic representation should transition accordingly.

That is **much more powerful than training only whole-utterance emotions**.

---

# 12. What about blends?

For:

```text
[playful=0.7,teasing=0.5]
```

we construct:

```text
style =
0.7 * playful_embedding
+
0.5 * teasing_embedding
```

and use that as the style representation for the entire segment.

Then your dataset teaches examples like:

```text
[playful=0.7] ...
[teasing=0.7] ...
[playful=0.7,teasing=0.5] ...
```

The model sees the individual styles and their combinations.

This is why the **3,000 blended examples in your bank are valuable**.

---

# 13. What I would NOT do

I wouldn't simply append the tags to the text and hope StyleTTS2 learns them.

For example:

```text
❌ text = "[sarcastic=0.8] Oh, great."
```

fed unchanged through the normal phonemizer.

The model could learn:

> `[sarcastic]` is some weird lexical pattern.

That's not a reliable style-control mechanism.

Instead:

```text
✅ raw input
[sarcastic=0.8] Oh, great.

        ↓ parser

text:
Oh, great.

style:
sarcastic → 0.8
```

Then style gets injected into the appropriate part of the acoustic model.

---

# 14. The customized KionTTS architecture

I'd aim conceptually for:

```text
                         KionTTS
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
          Text encoder              Style encoder
               │                         │
               │               ┌─────────┴─────────┐
               │               │                   │
               │          style embedding      intensity
               │               │                   │
               │               └────────┬──────────┘
               │                        │
               └──────────┬─────────────┘
                          ▼
                   StyleTTS2 core
                          │
                          ▼
                  acoustic generation
                          │
                          ▼
                    Kion waveform
```

And importantly:

```text
Kion speaker = fixed
```

There is no:

```text
speaker reference
speaker selection
voice cloning
```

at inference.

---

# 15. There is another decision we need to make

We have two ways to implement the style controls.

### Option A — style embedding conditioning

We explicitly create:

```text
emotion → learned embedding
```

This is simple and very controllable.

### Option B — learn a continuous style encoder

Instead of only having 37 explicit embeddings, we use the synthetic audio to learn a richer continuous representation and use your tags to guide/anchor that space.

That would be more aligned with StyleTTS2's philosophy of continuous latent style.

For Kion, **I'd lean toward a hybrid**:

```text
37 named style controls
        +
continuous learned latent style
```

So your LLM can explicitly request:

```text
sarcastic = 0.7
```

but the acoustic model still has enough freedom to produce natural variation instead of making every sarcastic sentence sound like the same preset.

---

# 16. Why your synthetic dataset needs variation

This is why I don't want:

```text
every sarcastic sample = alpha 0.8
every playful sample = alpha 0.7
```

We want:

```text
sarcastic 0.4
sarcastic 0.5
sarcastic 0.6
sarcastic 0.7
sarcastic 0.8
...
```

and different text.

Then the model can learn:

```text
emotion identity
        +
continuous strength
        +
linguistic context
        +
prosodic variation
```

instead of memorizing a fixed acoustic template.

---

# 17. Your current IndexTTS2 data can support this

Your desired training sample becomes:

```text
audio:
00018372.wav

text:
[playful=0.7,teasing=0.5] You really thought I wouldn't notice?

speaker:
Kion
```

During preprocessing:

```text
             raw metadata
                   │
           ┌───────┴────────┐
           ▼                ▼
          text             tags
           │                │
       phonemes         style vectors
           │                │
           └───────┬────────┘
                   ▼
              StyleTTS2
                   │
                   ▼
             target waveform
```

That's the core customization.

---

## And there is one major advantage for your specific project

Because **Kion is single-speaker**, the model doesn't have to simultaneously solve:

```text
Who is speaking?
What emotion are they expressing?
```

It only has to solve:

```text
Kion is speaking.
How should Kion say this?
```

That's a much cleaner learning problem.

Your IndexTTS2 generation stage provides the examples of **how Kion should sound under different expressive conditions**, while StyleTTS2 becomes the actual learned Kion voice.

### In other words

IndexTTS2:

> **"Here's an example of Kion sounding sarcastic."**

KionTTS:

> **"Got it. I learned what Kion-sarcastic sounds like. Now give me any new sentence and `[sarcastic=0.7]`, and I'll generate it myself."**

That's the whole point of the custom conditioning layer.

And before we start modifying StyleTTS2, I'd build a **small ~1k-sample prototype with maybe 6–8 of your strongest style tags**. If that prototype can reliably respond to `[emotion=intensity]`, transitions, and blends, then we scale the exact same architecture to your full 13k+ dataset.
