from typing import List

from .chatgpt import TranslationResponse
from .common import MessageRequest, FullTranslatedCaptionResult

TEST_MESSAGE: List[MessageRequest] = [
    MessageRequest(
        start="00:00:07.960",
        end="00:00:13.320",
        caption=[
            "Da den sidste istid som en softice",
            "smeltede for 11.000 år siden -"
        ]
    ),
    MessageRequest(
        start="00:00:13.440",
        end="00:00:18.320",
        caption=[
            " - så stak der fortsat",
            "400 danske øer op over vandlinjen."
        ]
    ),
    MessageRequest(
        start="00:00:18.440",
        end="00:00:23.840",
        caption=[
            "Nu får fire af dem besøg.",
            "Ærø, Fanø, Fur og Orø. "
        ]
    ),
]
TEST_RESPONSE = TranslationResponse(
    captions=[
        FullTranslatedCaptionResult(
            start="00:00:07.960",
            end="00:00:13.320",
            original=[
                "Da den sidste istid som en softice",
                "smeltede for 11.000 år siden -"
            ],
            translated=[
                'When the last ice age, like a soft serve ice cream, melted 11,000 years ago -'
            ]
        ),
        FullTranslatedCaptionResult(
            start="00:00:07.960",
            end="00:00:13.320",
            original=[
                ' - så stak der fortsat',
                '400 danske øer op over vandlinjen.'
            ],
            translated=[
                ' - 400 Danish islands still rose above the waterline.'
            ]
        ),
        FullTranslatedCaptionResult(
            start="00:00:18.440",
            end="00:00:23.840",
            original=[
                "Nu får fire af dem besøg.",
                "Ærø, Fanø, Fur og Orø. "
            ],
            translated=[
                'Now, four of them are being visited.',
                'Ærø, Fanø, Fur, and Orø.'
            ]
        )
    ])
