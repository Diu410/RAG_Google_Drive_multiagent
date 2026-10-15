"""One prompt budget for chat, document QA, workspace and intermediate agents."""
from llm_contract import ProviderError

# The application puts optional source material after these explicit labels.
# The question and instructions before a label are never silently shortened.
SOURCE_LABELS = ('\nКонтекст:\n', '\nКонтекст: ', '\n\nИсточники:\n',
                 '\nИсточники (единственный источник фактов):\n', '\n\nФайл ',
                 '\nЧерновик: ', '\nТекст: ', '\nОтредактированный: ')


def fit_prompt(messages, context_size, output_tokens, count):
    messages = [dict(item) for item in messages]
    output_tokens = min(output_tokens, context_size - 128)
    if output_tokens <= 0:
        raise ProviderError('Model context is too small for a response')
    input_limit = context_size - output_tokens - 32
    optional = []
    for index, item in enumerate(messages):
        content = item.get('content', '')
        if not isinstance(content, str):
            raise ProviderError('Only text messages are supported by this local model')
        positions = [(content.find(label), label) for label in SOURCE_LABELS if label in content]
        if positions:
            position, label = min(positions)
            end = position + len(label)
            optional.append((index, content[:end], content[end:]))
    # First discard old conversation turns. Sources and the current question
    # have higher priority than stale assistant replies.
    def token_count():
        return count(messages)

    tokens = token_count()
    while tokens > input_limit and len(messages) > 2:
        history_index = next((i for i, item in enumerate(messages[:-1])
                              if item['role'] != 'system'), None)
        if history_index is None:
            break
        messages.pop(history_index)
        optional = [(i - (i > history_index), prefix, source)
                    for i, prefix, source in optional if i != history_index]
        tokens = token_count()
    while tokens > input_limit and any(source for _, _, source in optional):
        # Jump closer to the budget for very oversized prompts. The estimate
        # only chooses a cut point; acceptance still uses the real tokenizer.
        largest = max(range(len(optional)), key=lambda i: len(optional[i][2]))
        index, prefix, source = optional[largest]
        target = min(len(source) // 2, int(len(source) * input_limit / max(tokens, 1)))
        retained = source[:target]
        optional[largest] = (index, prefix, retained)
        messages[index]['content'] = prefix + retained
        tokens = token_count()
    if tokens > input_limit:
        raise ProviderError('The question and instructions exceed the model context. Shorten the request '
                            'or configure a model with a larger context window.')
    return messages, output_tokens
