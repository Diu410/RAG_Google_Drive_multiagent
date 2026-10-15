"""Select installed local chat weights while retaining inference profiles."""
import json
import os

from assistant_models import AssistantError
from llm_contract import ProviderError
from settings import DATA_DIR
import model_profiles


def selection_status():
    selected = model_profiles.get_profile().model_path.resolve()
    choices = [{'id': model_id, 'label': label, 'available': path.is_file()}
               for model_id, label, path in model_profiles.chat_model_catalog()]
    selected_id = next((model_id for model_id, _, path in model_profiles.chat_model_catalog()
                        if path.resolve() == selected), None)
    return {'chat_models': choices, 'selected_model_id': selected_id}


def switch_chat_model(model_id, provider):
    choice = next((path for key, _, path in model_profiles.chat_model_catalog() if key == model_id), None)
    if choice is None:
        raise AssistantError('Неизвестная модель.', 422)
    if not choice.is_file():
        raise AssistantError('Файл выбранной модели не установлен.', 404)
    if choice.resolve() == model_profiles.get_profile().model_path.resolve():
        return selection_status()
    if provider.manager is None:
        raise AssistantError('Переключение доступно для управляемой локальной модели.', 409)
    previous_profiles = dict(model_profiles.PROFILES)
    previous_profile = provider.manager.profile
    try:
        with provider.deployment_change():
            destination = DATA_DIR / 'chat-model-selection.json'
            previous_selection = destination.read_bytes() if destination.exists() else None
            temporary = destination.with_suffix('.json.tmp')
            provider.manager.prepare_replacement()
            committed = False
            try:
                try:
                    temporary.write_text(json.dumps({'model_id': model_id}) + '\n', encoding='utf-8')
                    os.replace(temporary, destination)
                    committed = True
                finally:
                    temporary.unlink(missing_ok=True)
                model_profiles.select_chat_model_path(choice)
                provider.manager.profile = model_profiles.get_profile()
                provider.manager.error = None
                provider.manager.start_background()
            except BaseException:
                if committed and previous_selection is None:
                    destination.unlink(missing_ok=True)
                elif committed:
                    temporary.write_bytes(previous_selection)
                    os.replace(temporary, destination)
                model_profiles.PROFILES.clear()
                model_profiles.PROFILES.update(previous_profiles)
                provider.manager.profile = previous_profile
                provider.manager.start_background()
                raise
    except ProviderError as exc:
        raise AssistantError(str(exc), 409) from exc
    return selection_status()
