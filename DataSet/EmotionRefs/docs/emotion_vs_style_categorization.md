# Emotion vs. Style Categorization

To properly structure Kion's training metadata, we are strictly separating the inline tags into **Emotions** (the affective, internal state) and **Styles** (the external delivery or tone). 

This allows the final TTS model to learn independent control over *what* Kion is feeling versus *how* he is expressing it.

## 1. Emotions (Affective States)
These tags represent internal feelings and emotional reactions.

- **angry** 
- **annoyed**
- **bored**
- **concerned**
- **confused**
- **curious**
- **disappointed**
- **excited**
- **frustrated**
- **happy**
- **heartbroken** (normalized from `heartbreak`)
- **overjoyed**
- **sad**
- **surprised** (normalized from `suprised`)

## 2. Styles (Delivery / Prosody)
These tags represent the vocal tone, prosody, and delivery mechanics of the speech.

- **affectionate**
- **authoritative**
- **calm**
- **deadpan**
- **dramatic**
- **playful**
- **sarcasm** (normalized from `sarcastic`)
- **serious**
- **soothing**
- **teasing**

## Metadata Application
When the Colab generator processes an utterance like `[disappointed=0.7,sarcasm=0.6] Oh, brilliant.`, the resulting dataset metadata will no longer lump them into a single `styles` dictionary. 

Instead, it will use the categorization above to explicitly map:
```json
{
  "emotions": {
    "disappointed": 0.7
  },
  "styles": {
    "sarcasm": 0.6
  }
}
```

## 3. Available Single Emotion References (37)
The following are the available single emotion/style reference audio files in `voices/`:

- **Annoyed:** `voice_preview_annoyed-mild.wav`, `voice_preview_annoyed-strong.wav`
- **Angry:** `voice_preview_angry-mild.wav`, `voice_preview_angry-strong.wav`
- **Affectionate:** `voice_preview_affectionate.wav`
- **Authoritative:** `voice_preview_authoritative.wav`
- **Bored:** `voice_preview_bored.wav`
- **Calm:** `voice_preview_calm.wav`
- **Concerned:** `voice_preview_concerned-mild.wav`, `voice_preview_concerned-strong.wav`
- **Confused:** `voice_preview_confused-mild.wav`, `voice_preview_confused-strong.wav`
- **Curious:** `voice_preview_curious-mild.wav`, `voice_preview_curious-strong.wav`
- **Deadpan:** `voice_preview_deadpan.wav`
- **Disappointed:** `voice_preview_disappointed-mild.wav`, `voice_preview_disappointed-medium.wav`, `voice_preview_disappointed-strong.wav`
- **Dramatic:** `voice_preview_dramatic-medium.wav`, `voice_preview_dramatic-strong.wav`
- **Excited:** `voice_preview_excited-medium.wav`, `voice_preview_excited-strong.wav`
- **Frustrated:** `voice_preview_frustrated-mild.wav`, `voice_preview_frustrated-strong.wav`
- **Happy:** `voice_preview_happy-mild.wav`, `voice_preview_happy-strong.wav`
- **Heartbroken:** `voice_preview_heartbroken.wav`
- **Overjoyed:** `voice_preview_overjoyed.wav`
- **Playful:** `voice_preview_playful.wav`
- **Sad:** `voice_preview_sad-mild.wav`, `voice_preview_sad-strong.wav`
- **Sarcasm:** `voice_preview_dry-sarcasm.wav`
- **Serious:** `voice_preview_serious.wav`
- **Soothing:** `voice_preview_soothing.wav`
- **Surprised:** `voice_preview_surprised-mild.wav`, `voice_preview_surprised-strong.wav`
- **Teasing:** `voice_preview_teasing.wav`

## 4. Available Double Compound Emotion References (35)
- **Angry + Authoritative:** `voice_preview_angry0-7-authoritative0-8.wav`
- **Angry + Dramatic:** `voice_preview_angry0-8-dramatic0-8.wav`
- **Calm + Affectionate:** `voice_preview_calm0-7-affectionate0-6.wav`
- **Calm + Authoritative:** `voice_preview_calm0-8-authoritative0-7.wav`
- **Calm + Serious:** `voice_preview_calm0-8-serious0-7.wav`
- **Calm + Soothing:** `voice_preview_calm0-8-soothing0-7.wav`
- **Concerned + Affectionate:** `voice_preview_concerned0-7-affectionate0-6.wav`
- **Concerned + Serious:** `voice_preview_concerned0-7-serious0-7.wav`
- **Concerned + Soothing:** `voice_preview_concerned0-7-soothing0-7.wav`
- **Confused + Curious:** `voice_preview_confused0-6-curious0-7.wav`
- **Confused + Surprised:** `voice_preview_confused0-7-surprised0-6.wav`
- **Curious + Playful:** `voice_preview_curious0-7-playful0-6.wav`
- **Curious + Teasing:** `voice_preview_curious0-7-teasing0-6.wav`
- **Disappointed + Affectionate:** `voice_preview_disappointed0-6-affectionate0-6.wav`
- **Disappointed + Sarcasm:** `voice_preview_disappointed0-7-sarcasm0-6.wav`
- **Disappointed + Serious:** `voice_preview_disappointed0-7-serious0-7.wav`
- **Disappointed + Teasing:** `voice_preview_disappointed0-6-teasing0-5.wav`
- **Excited + Dramatic:** `voice_preview_excited0-8-dramatic0-8.wav`
- **Excited + Playful:** `voice_preview_excited0-8-playful0-7.wav`
- **Excited + Teasing:** `voice_preview_excited0-8-teasing0-6.wav`
- **Frustrated + Dramatic:** `voice_preview_frustrated0-7-dramatic0-8.wav`
- **Frustrated + Playful:** `voice_preview_frustrated0-6-playful0-5.wav`
- **Frustrated + Sarcasm:** `voice_preview_frustrated0-7-sarcasm0-7.wav`
- **Happy + Affectionate:** `voice_preview_happy0-7-affectionate0-7.wav`
- **Happy + Playful:** `voice_preview_happy0-7-playful0-6.wav`
- **Happy + Teasing:** `voice_preview_happy0-7-teasing0-6.wav`
- **Heartbroken + Affectionate:** `voice_preview_heartbroken0-9-affectionate0-6.wav`
- **Playful + Teasing:** `voice_preview_playful0-7-teasing0-7.wav`
- **Sad + Affectionate:** `voice_preview_sad0-7-affectionate0-6.wav`
- **Sad + Soothing:** `voice_preview_sad0-6-soothing0-7.wav`
- **Sarcasm + Playful:** `voice_preview_sarcasm0-7-playful0-6.wav`
- **Serious + Authoritative:** `voice_preview_serious0-8-authoritative0-8.wav`
- **Serious + Dramatic:** `voice_preview_serious0-8-dramatic0-7.wav`
- **Serious + Soothing:** `voice_preview_serious0-7-soothing0-7.wav`
- **Surprised + Dramatic:** `voice_preview_surprised0-8-dramatic0-7.wav`

## 5. Available Triple Compound Emotion References (4)
- **Concerned + Soothing + Affectionate:** `voice_preview_concerned0-7-soothing0-8-affectionate0-6.wav`
- **Disappointed + Sarcasm + Calm:** `voice_preview_disappointed0-7-sarcasm0-7-calm0-8.wav`
- **Frustrated + Sarcasm + Playful:** `voice_preview_frustrated0-7-sarcasm0-7-playful0-5.wav`
- **Happy + Affectionate + Teasing:** `voice_preview_happy0-7-affectionate0-6-teasing0-5.wav`
