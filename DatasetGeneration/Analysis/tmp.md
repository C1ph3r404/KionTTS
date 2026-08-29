## Overall verdict

I'd currently rate it around **7.5–8/10 as a behavioral SFT dataset**.

The architecture is strong. The biggest issue isn't lack of categories — it's that some behaviors are **too explicitly scripted**, especially around tools, emotions, and clarification.

And your recent observation about:

> “Is this code vulnerable?” → **“What code?”**

is exactly the kind of refinement this dataset needs.

---

# 1. The distribution is actually pretty solid

You currently have:

| Category                      | Examples |
| ----------------------------- | -------: |
| Contextual multi-turn         |      968 |
| Casual everyday               |      922 |
| Emotional / empathetic        |      725 |
| Coding / technical            |      599 |
| General knowledge             |      582 |
| Humor / banter                |      580 |
| Short reactions               |      503 |
| Kion personality              |      442 |
| Async / delayed calls         |      392 |
| Simple sync tools             |      387 |
| Initiative / proactive        |      329 |
| Disagreement / sarcasm        |      300 |
| Curiosity / uncertainty       |      296 |
| Multi-tool chained            |      291 |
| Tool + conversation           |      267 |
| Tool clarification            |      230 |
| Tool error recovery           |      167 |
| Correctly not using tool      |      158 |
| Cancellation/state management |      157 |
| Edge cases                    |      105 |

That's a **very good skeleton**.

The particularly good part is that you aren't just doing:

> personality + technical answers + tool calls

You've got things like:

* clarification
* cancellation
* asynchronous tasks
* errors
* state management
* initiative
* uncertainty
* knowing when **not** to use a tool
* multi-turn context

Those are exactly the behaviors that make an agent feel less like a chatbot.

---

# 2. Your tool training is one of the strongest parts

You have roughly **2,022 examples with tool calls** out of 8,400, so about **24%**.

That's substantial.

And I like that you have separate categories for:

> simple tool → chained tool → async → errors → cancellation → tool + conversation → don't use tool

That's much better than teaching:

```text
user asks something
↓
TOOL_CALL
↓
answer
```

You have examples like:

> user asks to start something
> tool runs
> task continues in background
> later system event arrives
> Kion reports completion

That's excellent for an agent.

### BUT...

There is a consistency problem.

Your tool representations aren't standardized.

I found things like:

```text
<tool_call>
get_time(location="Tokyo")
</tool_call>
```

but elsewhere:

```text
[TOOL_CALL: LaptopActionTool({"action": ...})]
```

and your JSON also has actual roles like:

```text
"role": "tool"
"role": "tool_result"
"role": "tool_call"
```

You need to decide **one canonical format** before the final 30k.

If your actual inference/training pipeline expects:

```text
assistant → tool call
tool → result
assistant → response
```

then your dataset should consistently represent that.

Don't make the model learn three different syntaxes for the same concept.

---

# 3. The biggest issue I see: Kion sometimes talks TOO MUCH

This is probably the biggest thing I'd fix.

You have a lot of responses like:

> “Checking active Elasticsearch tasks to identify the reindex task ID.”

or:

> “Tracking `migrate_events_2026` progress in the background.”

or:

> “GPG key configured. Writing the apt source entry and updating package indexes.”

These aren't necessarily bad.

But if your goal is **natural Kion**, you need to distinguish:

### Necessary communication

> “Checking.”

### Useful communication

> “The migration's still running.”

### Unnecessary narration

> “I am now proceeding to query the active Elasticsearch tasks in order to identify the relevant task ID.”

You don't want Kion narrating every internal action.

Especially with voice.

Imagine:

> **You:** “Check the logs.”

> **Kion:** “I'll check the logs now. First, I'll identify the relevant log files, then I'll inspect their timestamps and…”

Bro just **check the logs** 😭.

---

# 4. Your clarification behavior needs more examples

This is directly related to what we were discussing.

You have some good ones.

For example:

> “reschedule it to next week.”

Kion:

> **“reschedule which thing, and what day next week?”**

That's actually good.

But I'd expand this category **a lot**.

You currently have only **230 tool clarification examples**, and some of the dataset's clarification behavior is still slightly formal.

You need examples where Kion naturally responds to missing context with **2–6 word responses**.

For example:

```text
User: Fix this.

Kion: What's wrong?
```

```text
User: Is this vulnerable?

Kion: Which code?
```

```text
User: Run it.

Kion: Run what?
```

```text
User: Delete that.

Kion: Delete what?
```

```text
User: Change it.

Kion: Change what?
```

```text
User: Move it somewhere else.

Kion: Where?
```

```text
User: Send that to him.

Kion: Who?
```

```text
User: Do it tomorrow.

Kion: Do what?
```

That's **very valuable training data**.

Because you're teaching:

> missing information → **ask naturally**

rather than:

> missing information → produce an essay explaining why information is missing.

---

# 5. Your emotion system is VERY heavily represented

This jumped out immediately.

You have **34k+ Kion responses with emotion tags** out of ~38.5k Kion messages.

So approximately **90% of Kion responses are emotionally tagged**.

That's a lot.

And ~4,000 responses contain multiple emotion tags.

That's potentially problematic.

You don't want the model learning:

```text
Every response
↓
pick an emotion
↓
insert emotion tag
↓
speak
```

because then Kion may start sounding like:

> `[calm=0.6] yeah.`

> `[playful=0.5] okay.`

> `[happy=0.6] sure.`

> `[curious=0.5] interesting.`

every damn time.

Your TTS architecture needs the emotion tags, obviously. But **emotion tagging should represent actual expressive intent**, not become a mandatory grammatical component.

I'd intentionally increase **untagged/neutral responses**.

You currently have only around **27 explicitly marked `neutral/no tag` entries in your metadata**, although there are ~4k Kion messages without an emotion tag in the actual text.

That's an important distinction.

I'd preserve a substantial number of completely natural responses like:

> “yeah”

> “what?”

> “which one?”

> “send it”

> “okay”

> “that's weird”

> “nope”

> “probably”

> “I don't think so”

without forcing emotional decoration.

---

# 6. Your emotional combinations may be too aggressive

I saw stuff like:

```text
[frustrated=0.6, sarcasm=0.6]
```

and:

```text
[disappointed=0.5, teasing=0.7]
```

and:

```text
[authoritative=0.7, serious=0.8]
```

These can be good.

But **two emotions on ~4,000 responses** is something I'd monitor carefully.

The model could learn:

> complex answer = multiple emotions

instead of:

> emotion should correspond to conversational state.

I'd make single-emotion examples the dominant case, with combinations reserved for cases where the blend actually matters.

---

# 7. Your curiosity category is good, but I'd split it conceptually

You've got:

* conversational curiosity
* passive curiosity
* tool-enabled curiosity
* uncertainty

Good.

But I'd explicitly distinguish:

### Genuine curiosity

> “Wait, why'd you do it that way?”

### Clarification

> “Which file?”

### Uncertainty

> “I'm not sure.”

### Knowledge limitation

> “I don't know.”

### Tool decision

> “Let me check.”

### User-directed exploration

> “Want me to look into it?”

Those are **different behaviors**.

They're easy to accidentally collapse into one giant “curiosity” behavior.

---

# 8. Your initiative examples are especially valuable

329 examples isn't huge, but this is one of the categories I'd actually **increase**.

Because this is what separates:

> chatbot

from:

> agent.

For example:

```text
User:
The build failed.

Kion:
I'll check the error first.
```

versus:

```text
User:
The build failed.

Kion:
What do you want me to do?
```

The first demonstrates initiative.

But don't make initiative mean **always doing something without permission**.

You already have some explicitly authorized initiative, which is good.

I'd expand the distinction:

```text
Safe + obvious → act
Ambiguous → ask
Destructive → confirm
Needs information → investigate/tool
Potentially expensive → confirm
```

That's an extremely useful behavioral policy for Kion.

---

# 9. Your technical data is strong but sometimes TOO “expert monologue”

Some of the technical conversations are very good.

For example, the Rust/PostgreSQL chains demonstrate actual context retention.

But some responses are essentially textbook answers.

That's fine for technical capability, but your target isn't:

> Stack Overflow Kion.

It's:

> **Kion who happens to be technically capable.**

So I'd add more short conversational technical exchanges:

```text
User: why is this segfaulting?

Kion: show me the code.
```

```text
User: this query is slow

Kion: how slow?
```

```text
User: nginx is eating RAM

Kion: how much?
```

```text
User: the server keeps dying

Kion: logs?
```

That style is **very important**.

---

# 10. Your short-reaction category is good — expand it

503 examples.

I'd probably make this **800–1,000** in the final dataset.

Because short conversational responses teach the model that:

> **not every turn requires a paragraph.**

Examples:

> “yeah”

> “nah”

> “exactly”

> “wait, what?”

> “oh.”

> “fair.”

> “why?”

> “seriously?”

> “nice.”

> “damn.”

> “that's weird.”

> “which one?”

> “send it.”

> “go on.”

> “huh?”

These are disproportionately important for making a conversational model feel natural.

---

# 11. One thing I REALLY like: you have disagreement

300 disagreement/sarcasm examples.

Keep this.

A personality model that **always agrees with the user** becomes incredibly fake.

You want:

> User: “Python is obviously faster than C.”

Kion:

> “absolutely not.”

or:

> “that's... very optimistic.”

depending on personality.

But make sure disagreement isn't always sarcastic.

You need:

```text
gentle disagreement
neutral correction
firm disagreement
playful disagreement
technical correction
boundary disagreement
```

---

# 12. Your dataset has a hidden problem: synthetic regularity

This is probably the biggest concern once you go from 8.4k → 30k.

Your examples have a recognizable generated structure:

```text
User asks X
↓
Kion gives technically competent response
↓
emotion tag
↓
user responds
↓
Kion gives witty response
```

That's useful.

But if **30k examples follow this rhythm**, the model can learn the *shape* of your dataset rather than the behavior.

You need messy examples.

Real conversations aren't:

```text
Question
Answer
Question
Answer
Question
Answer
```

They contain:

> “wait”

> “nah”

> “actually…”

> “forget that”

> “no I meant…”

> “hold on”

> “what if…”

> “lol”

> typo

> correction

> incomplete sentence

> sudden topic switch

> follow-up referring to something 8 turns ago

> user changing their mind

You already have some of this.

**More.**

---

# 13. Dynamic thinking: your dataset isn't really teaching it yet

This is the area I'd add deliberately.

You currently have **curiosity/uncertainty**, but that's not the same thing as:

> **deciding whether reasoning is necessary.**

I'd add a dedicated behavioral category for **adaptive reasoning**.

Not:

```text
every difficult-looking question
→ <think>
```

Instead:

### Direct

```text
User: What's 25% of 800?

Kion: 200.
```

### Reasoning

```text
User: Three processes can acquire A and B in
different orders. Which execution can deadlock?

Kion:
<think>
...
</think>
...
```

### Clarification

```text
User: Is this vulnerable?

Kion: Which code?
```

### Tool

```text
User: Is the server reachable?

Kion:
<tool_call>...</tool_call>
```

### Uncertainty

```text
User: What was the exact temperature in this
city at 3:17 PM in 1998?

Kion:
I don't know.
```

The **decision itself** is what you're training.

---

# 14. There are also some factual-quality problems to audit

I spotted at least a few responses that are **too confidently specific**.

For example, your dataset contains technical recommendations with specific numbers such as:

> “A buffer of 10,000 to 50,000 items…”

Those kinds of values are highly context-dependent.

A model trained on them may learn:

> **technical question → confidently invent a precise number**

That's exactly the behavior you're trying to eliminate with uncertainty training.

For technical examples, I'd deliberately include:

```text
"depends on..."
"measure it first"
"I'd benchmark that"
"there isn't a universal value"
"I'd need to see..."
```

**when genuinely appropriate.**

Not as generic disclaimers — as actual technical judgment.

---

# 15. One structural issue: metadata isn't completely clean

You have:

* 8,400 rows
* 100 rows missing `doc_ref`
* 100 rows missing `has_tool_call`
* 817 without `sub_category`

That isn't catastrophic, but before finalizing:

**make metadata completely deterministic.**

If:

```text
has_tool_call = false
```

then it should actually correspond to the conversation.

Don't rely on metadata for training itself; use it for dataset analysis/filtering.

---

# What I'd do for the remaining ~21.6k examples

I **wouldn't simply generate another 21,600 examples proportionally**.

I'd use the remaining budget strategically.

Something like:

| New focus                           | Approx. additions |
| ----------------------------------- | ----------------: |
| Natural short conversation          |             2,500 |
| Context/memory across turns         |             2,000 |
| Clarification / missing context     |             1,800 |
| Adaptive reasoning                  |             1,800 |
| Uncertainty / epistemic behavior    |             1,500 |
| Initiative / agency                 |             1,500 |
| Tool selection / non-tool decisions |             1,500 |
| Messy human conversation            |             2,000 |
| Technical troubleshooting           |             1,500 |
| Tool error/recovery/state           |             1,500 |
| Personality / banter                |             1,500 |
| Disagreement / correction           |             1,000 |
| Emotional nuance                    |             1,500 |
| Edge cases                          |             1,000 |
| **Total**                           |       **~21,600** |

I would **not** spend the majority of the remaining examples on generic technical Q&A.

Your current dataset already has plenty of that.

---

# My biggest recommendations

If I were building this with you, the final 30k would prioritize:

### 🧠 1. Adaptive cognition

**Does Kion answer, think, ask, verify, or use a tool?**

### 🗣️ 2. Natural conversation

**Short, imperfect, contextual, occasionally messy.**

### ❓ 3. Natural clarification

Not:

> “I require additional information.”

But:

> **“Which one?”**

### 🤷 4. Genuine uncertainty

Not generic disclaimers.

### 🛠️ 5. Tool judgment

Not merely tool calling.

**Knowing when NOT to call a tool is equally important.**

### 🎭 6. Emotion as expression, not decoration

Don't make every sentence wear an `[emotion=0.7]` nametag.

### ⚡ 7. Response-length control

Sometimes:

> “yeah.”

Sometimes:

> 600 words.

The model should learn **why**.

---

## And honestly, your instinct from the last few messages is correct

You're starting to notice something important:

**The dataset shouldn't teach Kion to “behave like an AI assistant.”**

It should teach Kion to **behave like Kion.**

That means the target isn't:

> “How can I formulate the most helpful assistant response?”

It's more like:

> **“Given what's happening in this conversation, what's the most natural thing Kion would do next?”**

That's the mindset I'd use for the remaining 21.6k.

And **I would absolutely make adaptive reasoning + natural clarification + response-length control explicit categories before you generate the remaining data.** Those three are currently more valuable than just adding another 5,000 generic conversations.
