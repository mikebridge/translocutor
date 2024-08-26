from typing import List, Tuple

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from common import FullTranslatedCaptionResult, TranslatedCaptionResult, MessageRequest, SimpleLog
from text_utils import json_dump

# responses will contain a maximum of 4K tokens, regardless of input size,
# so we will estimate the result based on what we send, with a slight buffer.
DEFAULT_TARGET_TOKENS = 3500

# /**
# * In USD per 1000 tokens
# */
# export const ModelPricing = {
#     "gpt-3.5-turbo"             : { prompt: 0.50 / 1000000 * 1000,  completion: 1.50 / 1000000 * 1000 },
#     "gpt-3.5-turbo-1106"        : { prompt: 0.001,                  completion: 0.002 },
#     "gpt-4"                     : { prompt: 0.03,                   completion: 0.06 },
#     "gpt-4-32k"                 : { prompt: 0.06,                   completion: 0.12 },
#     "gpt-4-1106-preview"        : { prompt: 0.01,                   completion: 0.03 },
#     "gpt-4o"                    : { prompt: 5.00 / 1000000 * 1000,  completion: 15.00 / 1000000 * 1000 },
#     "gpt-4o-mini"               : { prompt: 0.150 / 1000000 * 1000, completion: 0.600 / 1000000 * 1000 },
# }
#
# export const ModelPricingAlias = {
#     "gpt-3.5-turbo-0301"        : "gpt-3.5-turbo",
#     "gpt-3.5-turbo-0613"        : "gpt-3.5-turbo",
#     "gpt-3.5-turbo-16k"         : "gpt-3.5-turbo-1106",
#     "gpt-3.5-turbo-16k-0613"    : "gpt-3.5-turbo-1106",
#     "gpt-4-0613"                : "gpt-4",
#     "gpt-4o-2024-05-13"         : "gpt-4o",
#     "gpt-4o-mini-2024-07-18"    : "gpt-4o-mini"
# }


class UsageResult(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TranslationResponse(BaseModel):
    captions: list[TranslatedCaptionResult]
    usage: UsageResult


def translate_subtitles(
        target_language: str,
        model: str,
        message_list: List[MessageRequest],
        on_start: SimpleLog,
        on_end: SimpleLog
) -> Tuple[List[FullTranslatedCaptionResult], int, UsageResult]:
    """
    Translate a list of subtitles/captions to a target language.

    see: https://openai.com/index/introducing-structured-outputs-in-the-api/

    :param target_language: The target language, in any form that chatgpt will understand (e.g. "English")
    :param message_list: list of captions to be sent to chatgpt
    :param model: the chatgpt api model (e.g. "gpt-4o")
    :param on_start: a function to be called with a status message when starting to translate a partition
    :param on_end: a function to be called with a status message when finished translating a partition
    """
    partitioned_message_request_list, estimated_token_count = partition_message_request_list(
        message_list,
        DEFAULT_TARGET_TOKENS,
        model)

    partitioned_results: List[List[FullTranslatedCaptionResult]]
    usage_result: UsageResult
    partitioned_results, usage_result = translate_partitioned_message_list(
        partitioned_message_request_list,
        target_language,
        on_start,
        on_end
    )

    # join up all the partitioned responses
    return ([caption_result for partitioned_result in partitioned_results for caption_result in partitioned_result],
            estimated_token_count, usage_result)


def estimate_tokens(message_request_list: List[MessageRequest], model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(json_dump(message_request_list))
    return len(tokens)


def partition_message_request_list(
        message_request_list: List[MessageRequest],
        target_tokens: int,
        model: str
) -> (
        Tuple)[List[List[MessageRequest]], int]:
    """
    partition the list of messages into groups that are small enough to send to the chatgpt api

    Note: this doesn't seem to do a good job at estimating the tokens in the request, based on the response we get back

    :param message_request_list: the list of messages to partition and send
    :param target_tokens: the target number of tokens to be used for partitioning the response
    :param model: the chatgpt model
    """
    partitioned_message_request_list: List[List[MessageRequest]] = [[]]
    total_token_count = 0
    current_token_count = 0
    for message_request in message_request_list:
        incremental_token_count = estimate_tokens([message_request], model)
        if current_token_count + incremental_token_count > target_tokens:
            partitioned_message_request_list.append([])
            current_token_count = 0
        current_token_count += incremental_token_count
        total_token_count += incremental_token_count
        partitioned_message_request_list[-1].append(message_request)
    return partitioned_message_request_list, total_token_count


def translate_partitioned_message_list(
        partitioned_message_request_list,
        target_language,
        on_start,
        on_end
) -> (
        Tuple)[List[List[FullTranslatedCaptionResult]], UsageResult]:
    client = OpenAI()
    partitioned_results: List[List[FullTranslatedCaptionResult]] = []
    total_usage_result: UsageResult = UsageResult(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0
    )
    for index, message_request_list in enumerate(partitioned_message_request_list, start=1):
        partitioned_caption_pair_list: List[FullTranslatedCaptionResult]
        usage: UsageResult

        on_start(f'translating partition {index} of {len(partitioned_message_request_list)}...')
        partitioned_caption_pair_list, usage = translate_messages(
            client,
            target_language,
            message_request_list)
        on_end("completed...")

        partitioned_results.append(partitioned_caption_pair_list)
        if total_usage_result and usage:
            total_usage_result = UsageResult(
                prompt_tokens=total_usage_result.prompt_tokens + usage.prompt_tokens,
                completion_tokens=total_usage_result.completion_tokens + usage.completion_tokens,
                total_tokens=total_usage_result.total_tokens + usage.total_tokens
            )
    return partitioned_results, total_usage_result


def translate_messages(client, lang, message_request_list) -> Tuple[List[FullTranslatedCaptionResult], UsageResult]:
    json_translation_message = json_dump(message_request_list)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": f"Translate the following sets of text to {lang}"},
            {"role": "user", "content": json_translation_message}
        ],
        response_format=TranslationResponse,
    )
    message = completion.choices[0].message
    # TODO: test the throwing of this
    if not message.parsed:
        # throw the error
        raise RuntimeError(f"OpenAI API call failed: {message.refusal}")
    translated_captions: List[TranslatedCaptionResult] = message.parsed.captions
    usage: UsageResult = message.parsed.usage

    # add the original text back in to each caption
    full_translated_captions: List[FullTranslatedCaptionResult] = []
    for index, caption in enumerate(translated_captions):
        full_translated_captions.append(FullTranslatedCaptionResult(
            start=caption.start,
            end=caption.end,
            original=message_request_list[index].caption,
            translated=caption.translated
        ))

    return full_translated_captions, usage
