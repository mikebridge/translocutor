#!/usr/bin/env python3
import argparse
import os
import pathlib
import sys

from dotenv import load_dotenv
from typing import TypedDict, List
from .chatgpt import translate_subtitles, UsageResult
from .common import MessageRequest, TranslatedCaptionResult
from .vtt import write_output_file, read_captions


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


def simple_log(message: str) -> None:
    print(message)


def process(main_args: MainArgs):
    message_list: List[MessageRequest] = read_captions(main_args['file'])
    # create a function that I can pass in that takes a string and prints it

    # caption_pair_list: List[CaptionResult] = TEST_RESPONSE.captions
    caption_pair_list: List[TranslatedCaptionResult]
    usage_result: UsageResult
    caption_pair_list, estimated_token_count, usage_result = translate_subtitles(
        main_args['target_language'],
        main_args['model'],
        message_list,
        simple_log,
        simple_log
    )

    # TODO: this number seems way off compared with what comes back from the result
    print('estimated tokens: ', estimated_token_count)
    new_file_name = create_file_name(main_args["file"]) or 'new_captions.vtt'
    print(f'writing to {new_file_name}')
    write_output_file(new_file_name, caption_pair_list)
    print('token usage:')
    print('    prompt:     ', usage_result.prompt_tokens)
    print('    completion: ', usage_result.completion_tokens)
    print('    total:      ', usage_result.total_tokens)


def main():
    load_dotenv()
    check_env_vars_or_exit()
    args = get_args()

    for file_path in args.file:
        print(f"reading file: {file_path}")
        main_args = MainArgs(
            file=file_path,
            model="gpt-4o",
            target_language=args.target_language)
        process(main_args)


if __name__ == '__main__':
    main()
