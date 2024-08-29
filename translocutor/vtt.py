import io
from typing import List

import webvtt
from webvtt import WebVTT

from .common import TranslatedCaptionResult, MessageRequest


STYLE = "position:10%,line-left align:left size:35%"
SYSTEM_CONTENT = "You are a helpful assistant that translates text."
COLOUR_ORIG = "yellow"
COLOUR_TRANSLATED = "white"


def read_vtt(file: str) -> WebVTT:
    caption_set: WebVTT = webvtt.read(file)
    return caption_set


def vtt_to_message(caption_sets: WebVTT) -> List[MessageRequest]:
    # TODO: implement this
    return [MessageRequest(
        start=caption.start,
        end=caption.end,
        caption=caption.text.split('\n')
    ) for caption in caption_sets]


def format_caption(colour: str, captions: List[str]) -> str:
    return f"<c.{colour}>{'\n'.join(captions)}</c>"


def create_vtt(caption_pair_list):
    vtt = webvtt.WebVTT()
    result: TranslatedCaptionResult
    for result in caption_pair_list:
        orig_caption = format_caption(COLOUR_ORIG, result.original)
        trans_caption = format_caption(COLOUR_TRANSLATED, result.translated)
        combined_caption = webvtt.Caption(result.start, f'{result.end} {STYLE}', [orig_caption, trans_caption])
        vtt.captions.append(combined_caption)
    return vtt


def write_output_file(new_file_name: str, caption_pair_list: List[TranslatedCaptionResult]):
    vtt = create_vtt(caption_pair_list)
    with io.open(new_file_name, 'w', encoding='utf8') as fd:
        vtt.write(fd)


def read_captions(file_name: str):
    captions = read_vtt(file_name)
    message_list = vtt_to_message(captions)
    return message_list
