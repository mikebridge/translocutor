import orjson


def json_dump(message_list):
    translation_message_bytes = orjson.dumps([message.model_dump() for message in message_list])
    json_translation_message: str = translation_message_bytes.decode('utf-8')
    return json_translation_message
