"""Skill contracts, discovery metadata and dependency-aware execution."""
from __future__ import annotations
import re
from typing import Any, Protocol, TYPE_CHECKING
from assistant_models import AssistantError, ToolResult
from agent_planner import AgentPlan, AgentPlanStatus

if TYPE_CHECKING:
    from assistant_core import ExecutionContext


class Skill(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    permissions: tuple[str, ...]

    async def execute(self, inputs: dict[str, Any], execution: 'ExecutionContext') -> ToolResult: ...


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        if skill.name in self._skills:
            raise ValueError(f'Skill already registered: {skill.name}')
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise AssistantError(f'Навык `{name}` не зарегистрирован.', 422) from exc

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def catalog(self) -> list[dict[str, Any]]:
        return [{
            'name': skill.name,
            'description': skill.description,
            'input_schema': skill.input_schema,
            'permissions': list(skill.permissions),
            'operations': list(getattr(skill, 'operations', ('execute',))),
            'operation_contracts': getattr(skill, 'operation_contracts', {}),
        } for skill in self.list()]


_REFERENCE = re.compile(r'^\$\{([a-z][a-z0-9_]{0,63})\.data\.([A-Za-z0-9_.-]+)\}$')


class AgentExecutor:
    """Executes a validated plan and passes bounded structured results between steps."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    @staticmethod
    def _resolve(value: Any, results: dict[str, ToolResult]) -> Any:
        if isinstance(value, str):
            match = _REFERENCE.fullmatch(value)
            if not match:
                return value
            step_id, path = match.groups()
            if step_id not in results:
                raise AssistantError(f'Результат шага `{step_id}` ещё недоступен.', 422)
            current: Any = results[step_id].data
            for part in path.split('.'):
                if not isinstance(current, dict) or part not in current:
                    raise AssistantError(f'Шаг `{step_id}` не вернул поле `{path}`.', 422)
                current = current[part]
            return current
        if isinstance(value, list):
            return [AgentExecutor._resolve(item, results) for item in value]
        if isinstance(value, dict):
            return {key: AgentExecutor._resolve(item, results) for key, item in value.items()}
        return value

    async def execute(self, plan: AgentPlan, execution, base_inputs: dict[str, Any] | None = None,
                      completed_results: dict[str, dict[str, Any]] | None = None,
                      confirmed_steps: set[str] | None = None) -> ToolResult:
        results: dict[str, ToolResult] = {
            step_id: ToolResult(success=True, tool='resumed', summary='Restored step', data=data)
            for step_id, data in (completed_results or {}).items()
        }
        confirmed_steps = confirmed_steps or set()
        final_parts: list[str] = []
        shared = dict(base_inputs or {})
        async with execution.action('PLAN_AGENT', {'objective': plan.objective}) as action:
            action.output = {'steps': [step.model_dump(mode='json') for step in plan.steps]}
        for step in plan.steps:
            if step.id in results:
                if results[step.id].data.get('final'):
                    final_parts.append(str(results[step.id].data['final']))
                continue
            if step.requires_confirmation and step.id not in confirmed_steps:
                return ToolResult(success=True, tool='agent', summary='Waiting for user confirmation', data={
                    'final': f'Для шага «{step.operation}» требуется подтверждение.',
                    'strategy': 'AGENT_WAITING_FOR_USER', 'plan_status': AgentPlanStatus.WAITING_FOR_USER,
                    'pending_step': step.model_dump(mode='json'), 'results': [], 'context': '',
                    'resume_state': {
                        'plan': plan.model_dump(mode='json'),
                        'completed_results': {key: value.data for key, value in results.items()},
                        'pending_step_id': step.id,
                        'base_inputs': shared,
                    },
                })
            arguments = dict(shared)
            arguments.update({key: value for key, value in self._resolve(step.arguments, results).items()
                              if not key.startswith('_') and key not in {'session_id', 'chat_id', 'workspace_id'}})
            skill = self.registry.get(step.skill)
            execute_step = getattr(skill, 'execute_agent_step', None)
            if step.operation == 'execute':
                result = await skill.execute(arguments, execution)
            elif execute_step:
                result = await execute_step(step.operation, arguments, execution)
            else:
                raise AssistantError(
                    f'Навык `{step.skill}` не поддерживает операцию `{step.operation}`.', 422
                )
            if not result.success:
                raise AssistantError(result.error or f'Шаг `{step.id}` завершился ошибкой.')
            results[step.id] = result
            if result.data.get('final'):
                final_parts.append(str(result.data['final']))
        last = results[plan.steps[-1].id]
        data = dict(last.data)
        data['final'] = '\n\n'.join(final_parts)
        data.setdefault('context', '')
        data.setdefault('results', [])
        data.setdefault('strategy', 'AGENT_PLAN')
        for step in reversed(plan.steps[:-1]):
            previous = results[step.id].data
            if data.get('preserve_previous_strategy') and previous.get('strategy'):
                data['strategy'] = previous['strategy']
                break
        data.pop('preserve_previous_strategy', None)
        data['plan_status'] = AgentPlanStatus.COMPLETED
        data['step_results'] = {key: value.data for key, value in results.items()}
        return ToolResult(success=True, tool='agent', summary='Agent plan completed', data=data)
