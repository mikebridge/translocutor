#!/usr/bin/env python3
import argparse
import os
import pathlib
import sys
import logging
import traceback

from dotenv import load_dotenv
from typing import TypedDict, List
from .chatgpt import translate_subtitles, UsageResult
from .common import MessageRequest, TranslatedCaptionResult
from .vtt import write_output_file, read_captions

# responses will contain a maximum of 4K tokens, regardless of input size,
# so we will estimate the result based on what we send, with a slight buffer.
DEFAULT_MAX_TOKENS = 3500


class MainArgs(TypedDict):
    file: str
    model: str
    target_language: str
    verbose: bool
    max_tokens: int


def create_file_name(orig_file_name: str, ext: str = 'all') -> str:
    p = pathlib.Path(orig_file_name)
    stem = p.stem
    orig_file_extension = p.suffix
    new_file_name = pathlib.Path(stem).stem
    return f'{new_file_name}.{ext}{orig_file_extension}'


def check_max_tokens(value):
    ivalue = int(value)
    if ivalue > 4000:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. The maximum allowed value is 4000.")
    return ivalue


def get_args():
    parser = argparse.ArgumentParser(description='Download Files from Google Drive')
    parser.add_argument('-f', '--file', nargs='+', help='vtt file(s)', required=True)
    parser.add_argument('-t', '--target-language', help='target language', default="English", required=False)
    parser.add_argument('-m', '--model', help='model (e.g. gpt-4o)', default="gpt-4o", required=False)
    parser.add_argument('-x', '--max-tokens', type=check_max_tokens,
                        help='maximum number of tokens per request, max 4000',
                        default=3500, required=False)
    parser.add_argument('-v', '--verbose', help='debugging output', action='store_true', default=False)

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


def process(main_args: MainArgs):
    message_list: List[MessageRequest] = read_captions(main_args['file'])
    # create a function that I can pass in that takes a string and prints it

    # caption_pair_list: List[CaptionResult] = TEST_RESPONSE.captions
    caption_pair_list: List[TranslatedCaptionResult]
    usage_result: UsageResult
    caption_pair_list, estimated_token_count, usage_result = translate_subtitles(
        main_args['target_language'],
        main_args['model'],
        main_args['max_tokens'],
        message_list,
        logging.info,
        logging.info
    )

    # TODO: this number seems way off compared with what comes back from the result
    logging.info('estimated tokens: %s', estimated_token_count)
    new_file_name = create_file_name(main_args["file"]) or 'new_captions.vtt'
    logging.info(f'writing to {new_file_name}')
    write_output_file(new_file_name, caption_pair_list)
    logging.info('token usage:')
    logging.info('    model:      %s', main_args['model'])
    logging.info('    prompt:     %s', usage_result.prompt_tokens)
    logging.info('    completion: %s', usage_result.completion_tokens)
    logging.info('    total:      %s', usage_result.total_tokens)


def main():
    load_dotenv()
    check_env_vars_or_exit()
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        for file_path in args.file:
            logging.info(f"reading file: {file_path}")
            logging.info('max tokens per request: %s', args.max_tokens)
            if args.verbose:
                logging.info('    verbose: True')
            main_args = MainArgs(
                file=file_path,
                model=args.model,
                target_language=args.target_language,
                verbose=args.verbose,
                max_tokens=args.max_tokens
            )
            process(main_args)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        if args.verbose:
            logging.debug("Traceback: %s", traceback.format_exc())
        else:
            logging.info("Run with --verbose for more information")
        sys.exit(1)


if __name__ == '__main__':
    main()
