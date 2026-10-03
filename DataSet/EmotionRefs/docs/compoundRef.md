| #  | Compound                            | Emotion/style intensities                       |
| -- | ----------------------------------- | ----------------------------------------------- |
| 1  | Happy + Playful                     | `happy=0.7, playful=0.6`                        |
| 2  | Happy + Teasing                     | `happy=0.7, teasing=0.6`                        |
| 3  | Playful + Teasing                   | `playful=0.7, teasing=0.7`                      |
| 4  | Happy + Affectionate                | `happy=0.7, affectionate=0.7`                   |
| 5  | Calm + Affectionate                 | `calm=0.7, affectionate=0.6`                    |
| 6  | Calm + Soothing                     | `calm=0.8, soothing=0.7`                        |
| 7  | Concerned + Soothing                | `concerned=0.7, soothing=0.7`                   |
| 8  | Concerned + Affectionate            | `concerned=0.7, affectionate=0.6`               |
| 9  | Sad + Soothing                      | `sad=0.6, soothing=0.7`                         |
| 10 | Sad + Affectionate                  | `sad=0.7, affectionate=0.6`                     |
| 11 | Heartbroken + Affectionate          | `heartbroken=0.9, affectionate=0.6`             |
| 12 | Disappointed + Sarcastic            | `disappointed=0.7, sarcasm=0.6`                 |
| 13 | Disappointed + Serious              | `disappointed=0.7, serious=0.7`                 |
| 14 | Disappointed + Teasing              | `disappointed=0.6, teasing=0.5`                 |
| 15 | Frustrated + Sarcastic              | `frustrated=0.7, sarcasm=0.7`                   |
| 16 | Frustrated + Dramatic               | `frustrated=0.7, dramatic=0.8`                  |
| 17 | Frustrated + Playful                | `frustrated=0.6, playful=0.5`                   |
| 18 | Angry + Dramatic                    | `angry=0.8, dramatic=0.8`                       |
| 19 | Angry + Authoritative               | `angry=0.7, authoritative=0.8`                  |
| 20 | Serious + Authoritative             | `serious=0.8, authoritative=0.8`                |
| 21 | Serious + Dramatic                  | `serious=0.8, dramatic=0.7`                     |
| 22 | Excited + Playful                   | `excited=0.8, playful=0.7`                      |
| 23 | Excited + Teasing                   | `excited=0.8, teasing=0.6`                      |
| 24 | Excited + Dramatic                  | `excited=0.8, dramatic=0.8`                     |
| 25 | Curious + Playful                   | `curious=0.7, playful=0.6`                      |
| 26 | Curious + Teasing                   | `curious=0.7, teasing=0.6`                      |
| 27 | Confused + Surprised                | `confused=0.7, surprised=0.6`                   |
| 28 | Surprised + Dramatic                | `surprised=0.8, dramatic=0.7`                   |
| 29 | Sarcastic + Playful                 | `sarcasm=0.7, playful=0.6`                      |
| 30 | Calm + Serious                      | `calm=0.8, serious=0.7`                         |
| 31 | Calm + Authoritative                | `calm=0.8, authoritative=0.7`                   |
| 32 | Concerned + Serious                 | `concerned=0.7, serious=0.7`                    |
| 33 | Confused + Curious                  | `confused=0.6, curious=0.7`                     |
| 34 | Disappointed + Affectionate         | `disappointed=0.6, affectionate=0.6`            |
| 35 | Serious + Soothing                  | `serious=0.7, soothing=0.7`                     |
| 36 | Sarcastic + Disappointed + Calm     | `sarcasm=0.7, disappointed=0.7, calm=0.8`       |
| 37 | Sarcastic + Frustrated + Playful    | `sarcasm=0.7, frustrated=0.7, playful=0.5`      |
| 38 | Concerned + Affectionate + Soothing | `concerned=0.7, affectionate=0.6, soothing=0.8` |
| 39 | Happy + Affectionate + Teasing      | `happy=0.7, affectionate=0.6, teasing=0.5`      |


No — **two references are enough to establish the two ends/directions of a 2-component compound, but not enough to reliably teach the whole 2D intensity space.**

For example:

```text
R1: disappointed=0.8, sarcasm=0.3
R2: disappointed=0.3, sarcasm=0.8
```

These give you two points:

```text
             sarcasm
                ↑
          R2 ●
             │
             │
             │
             │
             ● R1
                └────────→ disappointed
```

But what about:

```text
D=.8 S=.8
D=.2 S=.2
D=.8 S=.2
D=.2 S=.8
D=.6 S=.4
D=.4 S=.6
```

The model has not actually heard those performances.

### The good news

You **don't need a reference for every possible value**.

What I'd do is give each important 2-component compound **4 strategically chosen anchors**:

```text
                 sarcasm
                    ↑
        S=.8        ●────────●
                    │        │
                    │        │
        S=.3        ●────────●
                    └────────────→
                    D=.3     D=.8
```

For `disappointed + sarcasm`:

| Reference | Disappointed | Sarcasm |
| --------- | -----------: | ------: |
| A         |          0.8 |     0.3 |
| B         |          0.3 |     0.8 |
| C         |          0.8 |     0.8 |
| D         |          0.3 |     0.3 |

Now the model sees:

* disappointment-dominant
* sarcasm-dominant
* both strong
* both relatively weak

That's **far more informative**.

And you already have the original balanced reference, e.g.

```text
D=.7 S=.6
```

so that gives you a fifth point around the middle.

### But there's an important distinction

If you're relying on **IndexTTS2's `emo_alpha`**, that can give you variation in the *overall influence of that reference*. It does **not automatically mean the model learns independent `D` and `S` sliders**.

For example:

```text
Reference A
D=.8 S=.3
```

with:

```text
alpha = 0.3
alpha = 0.5
alpha = 0.7
alpha = 0.9
```

gives you different **strengths of that particular expressive mixture**.

It doesn't magically produce:

```text
D=.4 S=.7
```

because you've changed alpha.

---

### So for Kion I'd use this

For your **important 2-way compounds**:

**5 references total:**

```text
D=.3 S=.3
D=.8 S=.3
D=.3 S=.8
D=.8 S=.8
D=.6 S=.6   ← your existing balanced reference
```

Then use `emo_alpha` on each to generate additional overall-strength variation.

You don't need to do this for all 40 compounds immediately. Prioritize the combinations Kion is most likely to use:

```text
disappointed + sarcasm
sarcasm + playful
happy + playful
happy + teasing
concerned + soothing
concerned + affectionate
sad + soothing
frustrated + sarcasm
angry + authoritative
serious + authoritative
curious + playful
curious + teasing
```

That gives the model actual **2-dimensional coverage** instead of hoping it learns the entire space from two corner-ish examples.

And honestly, **this is the point where I'd stop adding references blindly and design the dataset mathematically first**. Otherwise we're going to end up with 200+ ElevenLabs references without knowing whether each one is actually contributing useful coverage.
