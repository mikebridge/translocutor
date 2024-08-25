from typing import List

from pydantic import BaseModel


#
# This will be response we tell OpenAI to return
#
class TranslatedCaptionResult(BaseModel):
    start: str
    end: str
    # original: List[str]
    translated: List[str]


#
# We'll then add the original rather than adding it to the payload that OpenAI will send back
#
class FullTranslatedCaptionResult(TranslatedCaptionResult):
    original: List[str]


#
# This is what we send to OpenAI
#
class MessageRequest(BaseModel):
    start: str
    end: str
    caption: List[str]  # list of lines in the caption
