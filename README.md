# AutoWordle

Playing **Wordle** (e.g. [Wordly](https://wordly.org)) is painful, especially when I was stuck with certain strange, unreasonable word and started to swear.

So instead of working hard to memorize words and practice Wordle, I cheat. This kind little script would play it for me; I can show off now.

This is a May Fool project.

## Setup

```bash
python -m venv .venv
# Source venv files for whatever you use
pip install -r requirements.txt
python autowordle.py
```

*READ THE CODE, EVERYTHING'S IN IT!*

If you are on Linux, you may have to use a system-wide Chrome installation to address permission issues. On Windows, nothing special has to be done - just alter `LOCAL_CHROME_PATH` to `None`.

## Usage

Now sit and watch. Star this repo if it helps.

This project is just for fun. Hack it if it doesn't work for you.

## Acknowledgement

- [A Collection of the Best Wordle Tips and Tricks](https://www.nytimes.com/2022/02/10/crosswords/best-wordle-tips.html) for the algorithm prototype.
- The project itself for allowing me to get rid of Wordle **forever**.
- Google Gemini, GitHub Copilot and Anthropic Claude for helping me refactor, simplify, modernize, modularize, commenting and altering to `playwright` instead of (deprecated) `pypuppet` my original **1,200-line** code.

## License

The Unlicense. Try your best to misuse it and tell me. (I'm not responsible for that.)
