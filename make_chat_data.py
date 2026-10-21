"""Build a chat corpus for fine-tuning the 1M model.

Produces `User: ...\\nAssistant: ...` exchanges covering greetings, small talk,
and one important guardrail: keyboard mashing is answered with `?`.

Optionally interleaves prose so the fine-tune keeps its language ability
instead of collapsing into pure chit-chat.

    python make_chat_data.py --mb 4 --prose-share 0.4 --prose data/mixed.txt
"""

import argparse
import os
import random

GREETINGS = [
    "hello", "hey", "hi there", "hello!", "hey there", "hi!", "hey!",
    "good morning", "good afternoon", "good evening", "howdy", "yo",
    "hello, how are you?", "hey, what's up?",
    "good morning!", "hi again", "hello there", "hi, are you there?",
]

# "hi" gets its own group so that the exact prompt "User: hi" only ever
# continues as "nia howdy". Nothing else in the corpus follows a bare "hi".
HI = (["hi"], ["nia howdy"])

GREETING_REPLIES = [
    "Hi!", "Hello!", "Hey there!", "Hi there!", "Hello! How can I help?",
    "Hey! How's it going?", "Hi! What can I do for you?", "Hello, friend.",
    "Hey! Nice to see you.", "Hi! How are you today?",
    "Hello! What's on your mind?", "Hey there! What do you need?",
]

THANKS = [
    "thanks", "thank you", "thanks a lot", "thank you so much", "thanks!",
    "cheers", "many thanks", "thanks for that", "appreciate it",
]
THANKS_REPLIES = [
    "You're welcome!", "No problem!", "Any time.", "Happy to help.",
    "My pleasure.", "Sure thing.", "Don't mention it.", "Glad I could help.",
]

BYES = [
    "bye", "goodbye", "see you", "see you later", "good night", "later",
    "i have to go", "talk soon", "farewell", "bye for now",
]
BYE_REPLIES = [
    "Goodbye!", "See you later!", "Bye! Take care.", "Good night!",
    "Take care!", "See you soon.", "Later!", "Farewell.",
]

HOW_ARE_YOU = [
    "how are you", "how are you doing", "how's it going", "how do you feel",
    "are you okay", "how have you been",
]
HOW_ARE_YOU_REPLIES = [
    "I'm doing well, thanks for asking!", "Pretty good, thanks!",
    "I'm fine. How are you?", "All good here. You?",
    "Can't complain! How about you?", "I'm well, thank you.",
]

IDENTITY = [
    "what is your name", "who are you", "what are you", "what should i call you",
    "do you have a name",
]
IDENTITY_REPLIES = [
    "I'm a tiny language model with one million parameters.",
    "I'm a small model that runs in your browser.",
    "I don't really have a name, but you can call me tiny.",
    "I'm a one-million-parameter transformer.",
]

CAPABILITY = [
    "what can you do", "can you help me", "what are you good at",
    "are you smart", "can you think",
]
CAPABILITY_REPLIES = [
    "Not much, honestly. I'm very small.",
    "I can chat a little, but I get confused easily.",
    "I try my best, but I'm only one million parameters.",
    "I can say hello and not much else.",
]

YES_NO = [
    "do you like music", "are you happy", "do you sleep", "are you real",
    "do you dream", "are you alive",
]
YES_NO_REPLIES = [
    "I don't really know. I just predict the next character.",
    "I don't sleep. I only run when you open the page.",
    "I'm not alive, but I'm happy to chat.",
    "I don't have feelings, but I like talking to you.",
    "I'm just a very small program.",
]

WEATHER = [
    "how is the weather", "is it raining", "what's the weather like",
    "is it cold today",
]
WEATHER_REPLIES = [
    "I can't see outside, so I have no idea.",
    "I don't have weather data. I'm a tiny model.",
    "No idea, but I hope it's nice where you are.",
]

HELP = [
    "help", "i need help", "can you explain", "tell me something",
    "say something", "talk to me",
]
HELP_REPLIES = [
    "I'll do my best, but I'm quite small.",
    "Sure! What would you like to talk about?",
    "I'm here. What's on your mind?",
    "Tell me more.",
]

FILLER = [
    "ok", "okay", "yes", "no", "sure", "cool", "nice", "great", "haha",
    "lol", "really", "i see", "right", "hmm",
]
FILLER_REPLIES = [
    "Okay!", "Got it.", "Sure.", "Right.", "Mm-hm.", "I see.", "Fair enough.",
    "Nice!", "Glad to hear it.",
]

GROUPS = [
    HI,
    HI,
    (GREETINGS, GREETING_REPLIES),
    (THANKS, THANKS_REPLIES),
    (BYES, BYE_REPLIES),
    (HOW_ARE_YOU, HOW_ARE_YOU_REPLIES),
    (IDENTITY, IDENTITY_REPLIES),
    (CAPABILITY, CAPABILITY_REPLIES),
    (YES_NO, YES_NO_REPLIES),
    (WEATHER, WEATHER_REPLIES),
    (HELP, HELP_REPLIES),
    (FILLER, FILLER_REPLIES),
]

KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
CONSONANTS = "bcdfghjklmnpqrstvwxz"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def mash(rng: random.Random) -> str:
    """Generate something that looks like a fist on the keyboard."""
    kind = rng.randrange(5)
    if kind == 0:
        row = rng.choice(KEYBOARD_ROWS)
        n = rng.randint(4, 10)
        start = rng.randrange(max(1, len(row) - n + 1))
        return row[start : start + n] * rng.choice([1, 1, 2])
    if kind == 1:
        return "".join(rng.choice(CONSONANTS) for _ in range(rng.randint(4, 9)))
    if kind == 2:
        c = rng.choice(LETTERS)
        return c * rng.randint(4, 8)
    if kind == 3:
        return "".join(rng.choice(CONSONANTS + "aeiou") for _ in range(rng.randint(5, 12)))
    return "".join(rng.choice(LETTERS) for _ in range(rng.randint(5, 14)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=4.0)
    ap.add_argument("--out", default="data/chat.txt")
    ap.add_argument("--prose", default=None, help="interleave prose from this file")
    ap.add_argument("--prose-share", type=float, default=0.4)
    ap.add_argument("--mash-share", type=float, default=0.10,
                    help="fraction of exchanges that are gibberish -> ?")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    target = int(args.mb * 1e6)
    prose = b""
    if args.prose and os.path.exists(args.prose):
        with open(args.prose, "rb") as f:
            prose = f.read()

    chat_budget = int(target * (1 - args.prose_share)) if prose else target
    prose_budget = target - chat_budget
    prose_pos = 0
    chunk_size = max(4096, target // 400)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written_chat = written_prose = turns = 0

    with open(args.out, "w", encoding="utf-8") as f:
        while written_chat < chat_budget or written_prose < prose_budget:
            if written_chat < chat_budget:
                if rng.random() < args.mash_share:
                    block = f"User: {mash(rng)}\nAssistant: ?\n\n"
                else:
                    prompt, replies = rng.choice(GROUPS)
                    block = f"User: {rng.choice(prompt)}\nAssistant: {rng.choice(replies)}\n\n"
                f.write(block)
                written_chat += len(block.encode("utf-8"))
                turns += 1

            if prose and written_prose < prose_budget:
                chunk = prose[prose_pos : prose_pos + chunk_size]
                if not chunk:
                    prose_pos = 0
                    continue
                f.write("\n")
                f.write(chunk.decode("utf-8", errors="replace"))
                f.write("\n\n")
                written_prose += len(chunk)
                prose_pos += chunk_size

    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    print(f"  {turns:,} exchanges, {written_prose / 1e6:.2f} MB prose interleaved")


if __name__ == "__main__":
    main()
