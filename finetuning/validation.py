from collections import defaultdict
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def validate_data_format(file_content):
    format_errors = defaultdict(int)

    for ex in file_content:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    return format_errors


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def validate_messages(dataset):
    convo_lens = []
    messages_errors = defaultdict(int)

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            messages_errors["n_missing_system"] += 1
        if not any(message["role"] == "user" for message in messages):
            messages_errors["n_missing_user"] += 1
        convo_lens.append(num_tokens_from_messages(messages))
    too_long = sum(len > 4096 for len in convo_lens)
    if too_long > 0:
        messages_errors["too_long"]
    return messages_errors
