from typing import List, Tuple

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from common import FullTranslatedCaptionResult, TranslatedCaptionResult, MessageRequest
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


class TranslationResponse(BaseModel):
    captions: list[TranslatedCaptionResult]


def estimate_tokens(message_request_list: List[MessageRequest], model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(json_dump(message_request_list))
    return len(tokens)


# TODO: verify that tiktoken is configured correctly for the payload
def translate_message_request_list(message_request_list: List[MessageRequest], target_tokens, model) -> (
        Tuple)[List[List[MessageRequest]], int]:
    partitioned_message_request_list: List[List[MessageRequest]] = [[]]
    total_token_count = 0
    current_token_count = 0
    for message_request in message_request_list:
        token_count = estimate_tokens([message_request], model)
        if current_token_count + token_count > target_tokens:
            partitioned_message_request_list.append([])
            current_token_count = 0
        current_token_count += token_count
        total_token_count += token_count
        partitioned_message_request_list[-1].append(message_request)
    return partitioned_message_request_list, total_token_count


def partition_and_translate_messages(
        target_language: str,
        model: str,
        message_list: List[MessageRequest]):

    partitioned_message_request_list, tokens = translate_message_request_list(
        message_list,
        DEFAULT_TARGET_TOKENS,
        model)

    print(f'estimated token count : {tokens}')
    partitioned_results = translate_partitioned_message_list(partitioned_message_request_list, target_language)

    # join up all the partitioned responses
    return [caption_result for partitioned_result in partitioned_results for caption_result in partitioned_result]


def translate_partitioned_message_list(partitioned_message_request_list, target_language):
    client = OpenAI()
    partitioned_results: List[List[TranslatedCaptionResult]] = []
    for index, message_request_list in enumerate(partitioned_message_request_list, start=1):
        print(f'translating partition {index} of {len(partitioned_message_request_list)}...')
        partitioned_caption_pair_list: List[FullTranslatedCaptionResult] = translate_messages(
            client,
            target_language,
            message_request_list)
        partitioned_results.append(partitioned_caption_pair_list)
    return partitioned_results


# see: https://openai.com/index/introducing-structured-outputs-in-the-api/
def translate_messages(client, lang, message_request_list) -> List[FullTranslatedCaptionResult]:
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
    if message.parsed:
        print('...completed')
        # print(message.parsed.captions)
    else:
        print("*** ERROR ***")
        print(message.refusal)
        exit(1)
    translated_captions: List[TranslatedCaptionResult] = message.parsed.captions

    # add the original text back in to each caption
    full_translated_captions: List[FullTranslatedCaptionResult] = []
    for index, caption in enumerate(translated_captions):
        full_translated_captions.append(FullTranslatedCaptionResult(
            start=caption.start,
            end=caption.end,
            original=message_request_list[index].caption,
            translated=caption.translated
        ))
    return full_translated_captions
