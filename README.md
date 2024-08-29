# Translocutor

Translocutor is a python CLI that will translate VTT subtitle files into another language 
using the ChatGPT `gpt-4o` api.  

Currently, it creates a single VTT file that contains the original subtitles along with
the translated language in a different colour.

This project is in an alpha state.

## Initial Setup

### OpenAI Key

First, ensure that you have an [API key for ChatGPT](https://platform.openai.com/docs/quickstart/create-and-export-an-api-key).

Then set it in your environment:

```sh
export OPENAI_API_KEY=my_key_value
```

You can alternatively set the key in the .env file.

### Python

It's recommended that you create a virtual python environment rather than use the 
default installation:

```sh
pip3 install venv
mkdir -i ~/venv # or wherever you want to store your virtual environments
cd ~/venv
python3 -m venv translocutor
source ~/venv/translocutor/bin/activate
```

Once you have a virtual environment active, you can install the required packages 
by running the following command:

```sh
pip install -r requirements.txt
```

Then set up the translocutor package in the venv:

```sh
pip install -e . 
```

## Usage

Once you've installed the required packages, you can run the script with the following command:

```
translocutor -f my_subtitle_file.fr.vtt
```

This will pass your subtitle file to the ChatGPT API, and write a new .vtt file

You can translate to something other than English (the default).  ChatGPT seems to be 
able to figure out what language you want.  Feel free to experiment.

```
translocutor -t german -f my_subtitle_file.fr.vtt
```

There is some minimal help available:

```
translocutor --help
```

## Getting VTT Files

[yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/Installation) can be a good way
of getting vtt files, e.g.:

```sh
yt-dlp --all-subs https://example.com/whatever-video
```

## Technical Notes

This uses ChatGPT API's [structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/).

Translocutor partitions the caption file and submits one partition at a time containing several lines.  
This should increase the accuracy of the translation by providing more context and continuity, 
instead of just submitting one line or one caption at a time.  This is done by estimating the 
tokens with [tiktoken](https://github.com/openai/tiktoken) and making a guess at the size of the 
result, to try to keep it under 4096 tokens.

## Legal Disclaimer

This project has nothing to do with ChatGPT or OpenAI.  It's just a project that uses their APIs.
