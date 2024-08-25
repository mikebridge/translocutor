# ChatGPT VTT Subtitle Translator

This is a python CLI that will translate VTT subtitle files into another language using the ChatGPT `gpt-4o` api.  

Currently, it creates a new VTT file with both the original language and then the translated language in 
a different colour.

This is in an alpha state.

## Initial Setup

### OpenAI Key

First, ensure that you have an [API key for ChatGPT](https://platform.openai.com/docs/quickstart/create-and-export-an-api-key).

Then set it in your environment, either by setting it or putting it in the `.env` file:

```sh
export OPENAI_API_KEY=my_key_value
```

### Python

It's recommended that you create a virtual python environment rather than use the default installation:

```sh
pip3 install venv
mkdir -i ~/venv # or wherever you want to store your virtual environments
cd ~/venv
python3 -m venv chatgpt-subtitle-translator
source ~/venv/chatgpt-subtitle-translator/bin/activate
```

Once you have a virtual environment active, you can install the required packages by running the following command:

```sh
pip install -r requirements.txt
```

Once you've installed the required packages, you can run the script with the following command:

```
python -m main -f my_subtitle_file.fr.vtt
```

This will pass your subtitle file to the ChatGPT API, and write a new .vtt file

You can translate to something other than English (the default).  ChatGPT seems to be 
able to figure out what language you want.  Feel free to experiment.

```
python -m main -t german -f my_subtitle_file.fr.vtt
```

There is some minimal help available:

```
python -m main --help
```

## Technical Notes

This uses the ChatGPT API and the 
