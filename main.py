#!/usr/bin/env python3
import argparse
import os
import pathlib
import sys

from dotenv import load_dotenv
from typing import TypedDict, List
from chatgpt import partition_and_translate_messages
from common import MessageRequest, TranslatedCaptionResult
from vtt import write_output_file, read_captions


class MainArgs(TypedDict):
    file: str
    model: str
    target_language: str


def create_file_name(orig_file_name: str, ext: str = 'all') -> str:
    p = pathlib.Path(orig_file_name)
    stem = p.stem
    orig_file_extension = p.suffix
    new_file_name = pathlib.Path(stem).stem
    return f'{new_file_name}.{ext}{orig_file_extension}'


def get_args():
    parser = argparse.ArgumentParser(description='Download Files from Google Drive')
    parser.add_argument('-f', '--file', nargs='+', help='vtt file(s)', required=True)
    parser.add_argument('-t', '--target-language', help='target language', default="English", required=False)
    parser.set_defaults(join=False)
    arg_list = parser.parse_args()
    for file in arg_list.file:
        if not os.path.exists(file):
            sys.stderr.write(f'ERROR: {file} does not exist\n')
            exit(1)
    return arg_list


#
# ensure that load_dotenv() is called before calling this.
#
def check_env_vars_or_exit():
    # we don't explicitly use this variable, but the openai api gets it from the environment,
    # so we check that it exists.
    open_api_key = os.getenv("OPENAI_API_KEY", None)
    if not open_api_key:
        sys.stderr.write(
            'please specify an OpenAI API key in the OPENAI_API_KEY environment variable (can be in .env).\n')
        exit(1)


def main(main_args: MainArgs):
    message_list: List[MessageRequest] = read_captions(main_args['file'])

    # caption_pair_list: List[CaptionResult] = TEST_RESPONSE.captions
    caption_pair_list: List[TranslatedCaptionResult] = partition_and_translate_messages(
        main_args['target_language'],
        main_args['model'],
        message_list)

    new_file_name = create_file_name(main_args["file"]) or 'new_captions.vtt'
    write_output_file(new_file_name, caption_pair_list)


if __name__ == '__main__':
    load_dotenv()
    check_env_vars_or_exit()
    args = get_args()

    for file_path in args.file:
        print(f"reading file: {file_path}")
        mainArgs = MainArgs(
            file=file_path,
            model="gpt-4o",
            target_language=args.target_language)
        main(mainArgs)
