try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import sys
import asyncio
import logging
import time as _time
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache

import httpx
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by the runtime.

# Load .env if present
load_dotenv()

# --- Logging Configuration ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

# --- Constants ---
BASE_URL = "https://workforce.example.com"

# --- Configuration Management ---
class Config:
    """Configuration loader for API keys, endpoints, and LLM credentials."""
    @staticmethod
    def get_oauth_token() -> Optional[str]:
        return os.getenv("WORKFORCE_API_TOKEN")

    @staticmethod
    def get_azure_openai_key() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @staticmethod
    def validate_for_api():
        """Validate that required API keys are present. Raises ValueError if not."""
        if not Config.get_oauth_token():
            raise ValueError("WORKFORCE_API_TOKEN is not configured in environment.")

    @staticmethod
    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_for_llm():
        """Validate that required LLM keys are present. Raises ValueError if not."""
        if not Config.get_azure_openai_key():
            raise ValueError("AZURE_OPENAI_API_KEY is not configured in environment.")
        if not Config.get_azure_openai_endpoint():
            raise ValueError("AZURE_OPENAI_ENDPOINT is not configured in environment.")
        if not Config.get_azure_openai_deployment():
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured in environment.")

# --- Utility Functions ---
@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(data: Any) -> Any:
    """Mask PII in logs and reports."""
    # For demo, just return data. Implement masking as needed.
    return data

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    """Sanitize input text for LLM and logs."""
    return text.strip().replace('\x00', '').replace('\r', '').replace('\x1a', '')

@with_content_safety(config=GUARDRAILS_CONFIG)
def format_error_response(message: str, tips: Optional[str] = None, error_type: str = "client_error") -> Dict[str, Any]:
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "tips": tips or ""
        }
    }

# --- Pydantic Models ---
class TaskModel(BaseModel):
    task_id: str = Field(..., min_length=1)
    required_skills: List[str] = Field(default_factory=list)
    estimated_effort: float = Field(..., gt=0)
    # Optionally, allow extra fields for LLM context

    @field_validator("task_id")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_task_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("task_id cannot be empty.")
        return v

    @field_validator("estimated_effort")
    @classmethod
    def validate_effort(cls, v):
        if v <= 0:
            raise ValueError("estimated_effort must be positive.")
        return v

class EmployeeModel(BaseModel):
    employee_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    # Optionally, allow extra fields for LLM context

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("employee_id cannot be empty.")
        return v

class AssignTasksRequest(BaseModel):
    tasks: List[TaskModel]
    employee_roster: List[EmployeeModel]

    @field_validator("tasks")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_tasks(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one task must be provided.")
        return v

    @field_validator("employee_roster")
    @classmethod
    def validate_roster(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one employee must be provided.")
        return v

class AssignTasksResponse(BaseModel):
    success: bool
    assignments: Optional[List[Dict[str, Any]]] = None
    utilization_summary: Optional[Dict[str, Any]] = None
    unassigned_tasks: Optional[List[Dict[str, Any]]] = None
    error: Optional[Dict[str, Any]] = None

class UtilizationSummaryRequest(BaseModel):
    date: Optional[str] = None

class UtilizationSummaryResponse(BaseModel):
    success: bool
    utilization_summary: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class UnassignedTasksRequest(BaseModel):
    date: Optional[str] = None

class UnassignedTasksResponse(BaseModel):
    success: bool
    unassigned_tasks: Optional[List[Dict[str, Any]]] = None
    error: Optional[Dict[str, Any]] = None

# --- API Client Base ---
class BaseAPIClient:
    """Base class for all API clients, handles auth and error handling."""
    def __init__(self):
        self.base_url = BASE_URL
        self._token = None

    def get_headers(self) -> Dict[str, str]:
        token = Config.get_oauth_token()
        if not token:
            raise ValueError("WORKFORCE_API_TOKEN is not configured.")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            _obs_t0 = _time.time()
            resp = await client.get(url, headers=headers, params=params)
            try:
                trace_tool_call(
                    tool_name='client.get',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            resp.raise_for_status()
            return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def _post(self, endpoint: str, json: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            _obs_t0 = _time.time()
            resp = await client.post(url, headers=headers, json=json)
            try:
                trace_tool_call(
                    tool_name='client.post',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(resp)[:200] if resp is not None else None,
                    status="success",
                )
            except Exception:
                pass
            resp.raise_for_status()
            return resp.json()

# --- API Clients ---
class AttendanceAPIClient(BaseAPIClient):
    async def get_attendance_status(self, employee_id: Optional[str] = None, date: Optional[str] = None) -> Any:
        params = {}
        if employee_id:
            params["employee_id"] = employee_id
        if date:
            params["date"] = date
        return await self._get("/api/attendance/status", params=params)

class SkillsAPIClient(BaseAPIClient):
    async def get_employee_skills(self, employee_id: Optional[str] = None) -> Any:
        params = {}
        if employee_id:
            params["employee_id"] = employee_id
        return await self._get("/api/employees/skills", params=params)

class CapacityAPIClient(BaseAPIClient):
    async def get_employee_capacity(self, employee_id: Optional[str] = None, date: Optional[str] = None) -> Any:
        params = {}
        if employee_id:
            params["employee_id"] = employee_id
        if date:
            params["date"] = date
        return await self._get("/api/employees/capacity", params=params)

class TaskAPIClient(BaseAPIClient):
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_priority(self, task_id: Optional[str] = None) -> Any:
        params = {}
        if task_id:
            params["task_id"] = task_id
        return await self._get("/api/tasks/priority", params=params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_due_date(self, task_id: Optional[str] = None) -> Any:
        params = {}
        if task_id:
            params["task_id"] = task_id
        return await self._get("/api/tasks/due-date", params=params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_dependencies(self, task_id: Optional[str] = None) -> Any:
        params = {}
        if task_id:
            params["task_id"] = task_id
        return await self._get("/api/tasks/dependencies", params=params)

class AssignmentAPIClient(BaseAPIClient):
    async def create_assignments(self, assignments: List[Dict[str, Any]]) -> Any:
        payload = {"assignments": assignments}
        return await self._post("/api/assignments", json=payload)

class UtilizationAPIClient(BaseAPIClient):
    async def get_utilization_summary(self, date: Optional[str] = None) -> Any:
        params = {}
        if date:
            params["date"] = date
        return await self._get("/api/assignments/utilization-summary", params=params)

class UnassignedTasksAPIClient(BaseAPIClient):
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_unassigned_tasks(self, date: Optional[str] = None) -> Any:
        params = {}
        if date:
            params["date"] = date
        return await self._get("/api/assignments/unassigned-tasks", params=params)

# --- Adapters ---
class ReportingToolAdapter:
    """Generates and formats assignment reports."""
    def generate_report(self, assignments, utilization_summary, unassigned_tasks) -> str:
        try:
            from jinja2 import Template
            template_str = """
            <h2>Assignment Report</h2>
            <h3>Assignments</h3>
            <ul>
            {% for a in assignments %}
              <li>Task {{a.task_id}}: {{a.employee_id}} (Allocation: {{a.allocation}})</li>
            {% endfor %}
            </ul>
            <h3>Utilization Summary</h3>
            <ul>
            {% for emp, util in utilization_summary.items() %}
              <li>{{emp}}: {{util}}%</li>
            {% endfor %}
            </ul>
            <h3>Unassigned Tasks</h3>
            <ul>
            {% for ut in unassigned_tasks %}
              <li>Task {{ut.task_id}}: {{ut.reason}}</li>
            {% endfor %}
            </ul>
            """
            template = Template(template_str)
            report = template.render(
                assignments=assignments or [],
                utilization_summary=utilization_summary or {},
                unassigned_tasks=unassigned_tasks or []
            )
            return report
        except Exception as e:
            logger.error(f"Report formatting error: {e}")
            return "Partial report generated due to formatting error."

class NotificationServiceAdapter:
    """Sends notifications to employees and managers."""
    async def notify_assignments(self, assignments: List[Dict[str, Any]]) -> bool:
        # Placeholder: In production, integrate with email/SMS/Slack/etc.
        logger.info(f"Notifying assignments: {mask_pii(assignments)}")
        await asyncio.sleep(0.1)
        return True

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def notify_unassigned_tasks(self, unassigned_tasks: List[Dict[str, Any]]) -> bool:
        logger.info(f"Notifying unassigned tasks: {mask_pii(unassigned_tasks)}")
        await asyncio.sleep(0.1)
        return True

# --- Audit Logger ---
class AuditLogger:
    """Logs all assignment decisions for audit and compliance."""
    def log_decision(self, decision_context: Dict[str, Any]):
        logger.info(f"Audit log: {mask_pii(decision_context)}")

# --- Business Rules Engine ---
class BusinessRulesEngine:
    """Evaluates and enforces all assignment business rules and decision tables."""
    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # Implement rule logic as per design
        if rule_id == "RULE_1":
            # Exclude Unavailable Employees
            if context.get("attendance_status") in ("absent", "leave"):
                return False, "Employee is absent or on leave"
            return True, None
        elif rule_id == "RULE_2":
            # Adjust Capacity for Half-Day
            if context.get("attendance_status") == "half-day":
                return True, "capacity_adjusted"
            return True, None
        elif rule_id == "RULE_3":
            # Match Required Skills
            emp_skills = set(context.get("employee_skills", []))
            task_skills = set(context.get("task_required_skills", []))
            if not task_skills.issubset(emp_skills):
                return False, "Skill mismatch"
            return True, None
        elif rule_id == "RULE_4":
            # Do Not Exceed Capacity
            if context.get("employee_capacity", 0) < context.get("task_estimated_effort", 0):
                return False, "Capacity exceeded"
            return True, None
        elif rule_id == "RULE_5":
            # Respect Task Dependencies
            if context.get("dependency_status") == "unresolved":
                return False, "Dependency unresolved"
            return True, None
        elif rule_id == "RULE_6":
            # Prioritize High Priority Tasks
            # Handled in orchestrator by sorting
            return True, None
        return True, None

    def apply_decision_table(self, table_id: str, context: Dict[str, Any]) -> str:
        # Implements DT_1: Assignment Eligibility Table
        if table_id == "DT_1":
            if context.get("attendance_status") != "present":
                return "ineligible"
            emp_skills = set(context.get("employee_skills", []))
            task_skills = set(context.get("task_required_skills", []))
            if not task_skills.issubset(emp_skills):
                return "ineligible"
            if context.get("employee_capacity", 0) < context.get("task_estimated_effort", 0):
                return "ineligible"
            return "eligible"
        return "eligible"

# --- LLM Integration (Azure OpenAI) ---
class AzureOpenAIClient:
    """Async Azure OpenAI client for LLM calls."""
    def __init__(self):
        self._client = None
        self._deployment = None
        self._endpoint = None
        self._api_key = None

    def _lazy_init(self):
        import openai
        api_key = Config.get_azure_openai_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        if not api_key or not endpoint or not deployment:
            raise ValueError("Azure OpenAI credentials are not fully configured.")
        self._api_key = api_key
        self._endpoint = endpoint
        self._deployment = deployment
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
        )

    def get_client(self):
        if self._client is None:
            self._lazy_init()
        return self._client

    @property
    def deployment(self):
        if self._deployment is None:
            self._lazy_init()
        return self._deployment

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def chat_completion(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        client = self.get_client()
        deployment = self.deployment
        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=deployment,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else ""
            )
        except Exception:
            pass
        return content

# --- Assignment Orchestrator ---
class AssignmentOrchestrator:
    """Coordinates the assignment process, invokes business rules, manages data retrieval and persistence."""
    def __init__(self):
        self.attendance_api = AttendanceAPIClient()
        self.skills_api = SkillsAPIClient()
        self.capacity_api = CapacityAPIClient()
        self.task_api = TaskAPIClient()
        self.assignment_api = AssignmentAPIClient()
        self.utilization_api = UtilizationAPIClient()
        self.unassigned_api = UnassignedTasksAPIClient()
        self.reporting_adapter = ReportingToolAdapter()
        self.notification_adapter = NotificationServiceAdapter()
        self.audit_logger = AuditLogger()
        self.rules_engine = BusinessRulesEngine()
        self.llm_client = AzureOpenAIClient()

        # LLM config
        self.system_prompt = (
            "You are an Employee Work Assignment Agent. Assign daily work only to employees marked as available according to attendance status. "
            "Use task priority, due date, required skills, dependencies, and individual capacity to create balanced allocations. "
            "Exclude absent or leave employees, reduce capacity for half-day availability, and output assignments, utilization summary, and clearly labeled unassigned tasks with reasons. "
            "Communicate in a professional, concise, and transparent manner."
        )
        self.user_prompt_template = "Please provide the list of tasks and employee roster for today's assignments."
        self.few_shot_examples = [
            "Assign the following tasks: Task A (high, due today, skill X), Task B (medium, due tomorrow, skill Y). Employees: John (present, skill X, full capacity), Jane (half-day, skill Y, half capacity), Mike (absent).\nAssignments:\n- Task A: John\n- Task B: Jane (capacity adjusted for half-day)\nUtilization Summary:\n- John: 100%\n- Jane: 50%\nUnassigned Tasks:\n- None",
            "Tasks: Task C (high, skill Z), Task D (low, skill Y). Employees: Alice (present, skill Y), Bob (leave).\nAssignments:\n- Task D: Alice\nUtilization Summary:\n- Alice: 50%\nUnassigned Tasks:\n- Task C: No available employee with required skill Z"
        ]
        self.temperature = 0.7
        self.max_tokens = 2000

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def assign_tasks(self, tasks: List[Dict[str, Any]], employee_roster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assigns tasks to employees based on all business rules and constraints."""
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Validate and normalize input data",
            output_fn=lambda r: f"tasks={len(tasks)}, employees={len(employee_roster)}"
        ) as step:
            # Input validation is already handled by Pydantic, but double-check
            if not tasks or not employee_roster:
                step.capture({"error": "No tasks or employees provided."})
                return format_error_response("No tasks or employees provided.", "Provide at least one task and one employee.")

            step.capture({"tasks": len(tasks), "employees": len(employee_roster)})

        # Fetch all required data in parallel
        async with trace_step(
            "fetch_data", step_type="tool_call",
            decision_summary="Fetch attendance, skills, capacity, and task details",
            output_fn=lambda r: f"attendance={len(r.get('attendance',{}))}, skills={len(r.get('skills',{}))}"
        ) as step:
            try:
                attendance_futures = [
                    self.attendance_api.get_attendance_status(emp["employee_id"])
                    for emp in employee_roster
                ]
                skills_futures = [
                    self.skills_api.get_employee_skills(emp["employee_id"])
                    for emp in employee_roster
                ]
                capacity_futures = [
                    self.capacity_api.get_employee_capacity(emp["employee_id"])
                    for emp in employee_roster
                ]
                task_priority_futures = [
                    self.task_api.get_task_priority(task["task_id"])
                    for task in tasks
                ]
                task_due_date_futures = [
                    self.task_api.get_task_due_date(task["task_id"])
                    for task in tasks
                ]
                task_dependencies_futures = [
                    self.task_api.get_task_dependencies(task["task_id"])
                    for task in tasks
                ]
                # Gather all
                (
                    attendance_list,
                    skills_list,
                    capacity_list,
                    priority_list,
                    due_date_list,
                    dependencies_list
                ) = await asyncio.gather(
                    asyncio.gather(*attendance_futures),
                    asyncio.gather(*skills_futures),
                    asyncio.gather(*capacity_futures),
                    asyncio.gather(*task_priority_futures),
                    asyncio.gather(*task_due_date_futures),
                    asyncio.gather(*task_dependencies_futures)
                )
                # Map employee_id/task_id to data
                attendance = {emp["employee_id"]: att for emp, att in zip(employee_roster, attendance_list)}
                skills = {emp["employee_id"]: skl for emp, skl in zip(employee_roster, skills_list)}
                capacity = {emp["employee_id"]: cap for emp, cap in zip(employee_roster, capacity_list)}
                priority = {task["task_id"]: pr for task, pr in zip(tasks, priority_list)}
                due_date = {task["task_id"]: dd for task, dd in zip(tasks, due_date_list)}
                dependencies = {task["task_id"]: dep for task, dep in zip(tasks, dependencies_list)}
                step.capture({
                    "attendance": attendance,
                    "skills": skills,
                    "capacity": capacity,
                    "priority": priority,
                    "due_date": due_date,
                    "dependencies": dependencies
                })
            except Exception as e:
                logger.error(f"API data fetch error: {e}")
                step.capture({"error": str(e)})
                return format_error_response("Failed to fetch employee/task data.", str(e), error_type="api_error")

        # Assignment logic
        async with trace_step(
            "assignment_logic", step_type="process",
            decision_summary="Apply business rules and assign tasks",
            output_fn=lambda r: f"assignments={len(r.get('assignments',[]))}, unassigned={len(r.get('unassigned_tasks',[]))}"
        ) as step:
            assignments = []
            unassigned_tasks = []
            utilization = {emp["employee_id"]: 0 for emp in employee_roster}
            # Prepare employee pool
            employee_pool = []
            for emp in employee_roster:
                emp_id = emp["employee_id"]
                att_status = attendance.get(emp_id, {}).get("status", "absent")
                # RULE_1: Exclude Unavailable Employees
                ok, reason = self.rules_engine.evaluate_rule("RULE_1", {"attendance_status": att_status})
                if not ok:
                    continue
                # RULE_2: Adjust Capacity for Half-Day
                cap = capacity.get(emp_id, {}).get("capacity", 0)
                if att_status == "half-day":
                    cap = cap * 0.5
                employee_pool.append({
                    "employee_id": emp_id,
                    "name": emp.get("name", ""),
                    "skills": skills.get(emp_id, {}).get("skills", []),
                    "capacity": cap,
                    "attendance_status": att_status
                })
            # Sort tasks by priority and due date (RULE_6)
            @trace_agent(agent_name='Employee Work Assignment Agent')
            @with_content_safety(config=GUARDRAILS_CONFIG)
            def task_sort_key(task):
                pr = priority.get(task["task_id"], {}).get("priority", "low")
                dd = due_date.get(task["task_id"], {}).get("due_date", "9999-12-31")
                pr_val = {"high": 1, "medium": 2, "low": 3}.get(pr, 3)
                return (pr_val, dd)
            sorted_tasks = sorted(tasks, key=task_sort_key)
            # Assign tasks
            for task in sorted_tasks:
                assigned = False
                for emp in employee_pool:
                    context = {
                        "attendance_status": emp["attendance_status"],
                        "employee_skills": emp["skills"],
                        "employee_capacity": emp["capacity"],
                        "task_required_skills": task.get("required_skills", []),
                        "task_estimated_effort": task.get("estimated_effort", 0)
                    }
                    # Decision table
                    eligibility = self.rules_engine.apply_decision_table("DT_1", context)
                    if eligibility != "eligible":
                        continue
                    # RULE_3: Match Required Skills
                    ok, reason = self.rules_engine.evaluate_rule("RULE_3", context)
                    if not ok:
                        continue
                    # RULE_4: Do Not Exceed Capacity
                    ok, reason = self.rules_engine.evaluate_rule("RULE_4", context)
                    if not ok:
                        continue
                    # RULE_5: Respect Task Dependencies
                    dep_status = "resolved"
                    deps = dependencies.get(task["task_id"], {}).get("dependencies", [])
                    if deps:
                        # For demo, assume all dependencies are resolved
                        dep_status = "resolved"
                    ok, reason = self.rules_engine.evaluate_rule("RULE_5", {"dependency_status": dep_status})
                    if not ok:
                        continue
                    # Assign
                    assignments.append({
                        "employee_id": emp["employee_id"],
                        "task_id": task["task_id"],
                        "allocation": task.get("estimated_effort", 0)
                    })
                    utilization[emp["employee_id"]] += task.get("estimated_effort", 0)
                    emp["capacity"] -= task.get("estimated_effort", 0)
                    assigned = True
                    break
                if not assigned:
                    # Find reason
                    reason = "No available employee"
                    for emp in employee_pool:
                        context = {
                            "attendance_status": emp["attendance_status"],
                            "employee_skills": emp["skills"],
                            "employee_capacity": emp["capacity"],
                            "task_required_skills": task.get("required_skills", []),
                            "task_estimated_effort": task.get("estimated_effort", 0)
                        }
                        eligibility = self.rules_engine.apply_decision_table("DT_1", context)
                        if eligibility != "eligible":
                            reason = "Eligibility rule failed"
                            break
                        ok, reason2 = self.rules_engine.evaluate_rule("RULE_3", context)
                        if not ok:
                            reason = reason2
                            break
                        ok, reason2 = self.rules_engine.evaluate_rule("RULE_4", context)
                        if not ok:
                            reason = reason2
                            break
                        ok, reason2 = self.rules_engine.evaluate_rule("RULE_5", {"dependency_status": dep_status})
                        if not ok:
                            reason = reason2
                            break
                    unassigned_tasks.append({
                        "task_id": task["task_id"],
                        "reason": reason
                    })
            # Normalize utilization to percent (assume full capacity is sum of assigned + remaining)
            utilization_summary = {}
            for emp in employee_pool:
                emp_id = emp["employee_id"]
                assigned = utilization.get(emp_id, 0)
                total = assigned + emp["capacity"]
                percent = int(100 * assigned / total) if total > 0 else 0
                utilization_summary[emp_id] = percent
            step.capture({
                "assignments": assignments,
                "utilization_summary": utilization_summary,
                "unassigned_tasks": unassigned_tasks
            })

        # Persist assignments
        async with trace_step(
            "persist_assignments", step_type="tool_call",
            decision_summary="Create assignments in external system",
            output_fn=lambda r: f"status={r.get('status','?')}"
        ) as step:
            try:
                if assignments:
                    result = await self.assignment_api.create_assignments(assignments)
                    step.capture({"status": "success", "result": result})
                else:
                    step.capture({"status": "no_assignments"})
            except Exception as e:
                logger.error(f"Assignment persistence error: {e}")
                step.capture({"error": str(e)})
                # Fallback: continue with in-memory assignments

        # LLM explanation
        async with trace_step(
            "llm_explanation", step_type="llm_call",
            decision_summary="Call LLM to generate assignment explanation",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                # Compose LLM prompt
                user_prompt = (
                    f"Today's tasks: {tasks}\n"
                    f"Employee roster: {employee_roster}\n"
                    f"Assignments: {assignments}\n"
                    f"Utilization Summary: {utilization_summary}\n"
                    f"Unassigned Tasks: {unassigned_tasks}\n"
                    "Explain the assignment decisions in a professional, concise, and transparent manner."
                )
                messages = [
                    {"role": "system", "content": self.system_prompt},
                ]
                for ex in self.few_shot_examples:
                    messages.append({"role": "user", "content": ex})
                messages.append({"role": "user", "content": user_prompt})
                explanation = await self.llm_client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                step.capture(explanation)
            except Exception as e:
                logger.error(f"LLM explanation error: {e}")
                explanation = "Assignment explanation unavailable due to LLM error."
                step.capture(explanation)

        # Audit log
        self.audit_logger.log_decision({
            "assignments": assignments,
            "utilization_summary": utilization_summary,
            "unassigned_tasks": unassigned_tasks,
            "explanation": explanation
        })

        return {
            "success": True,
            "assignments": assignments,
            "utilization_summary": utilization_summary,
            "unassigned_tasks": unassigned_tasks,
            "explanation": explanation
        }

    async def generate_utilization_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generates a summary of employee utilization after assignments."""
        async with trace_step(
            "generate_utilization_summary", step_type="tool_call",
            decision_summary="Fetch utilization summary from API",
            output_fn=lambda r: f"utilization={r}"
        ) as step:
            try:
                result = await self.utilization_api.get_utilization_summary(date)
                step.capture(result)
                return {"success": True, "utilization_summary": result}
            except Exception as e:
                logger.error(f"Utilization summary API error: {e}")
                # Fallback: return empty summary
                step.capture({"error": str(e)})
                return format_error_response("Failed to fetch utilization summary.", str(e), error_type="api_error")

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def output_unassigned_tasks(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Lists unassigned tasks with explicit reasons."""
        async with trace_step(
            "output_unassigned_tasks", step_type="tool_call",
            decision_summary="Fetch unassigned tasks from API",
            output_fn=lambda r: f"unassigned={len(r.get('unassigned_tasks',[])) if isinstance(r,dict) else '?'}"
        ) as step:
            try:
                result = await self.unassigned_api.get_unassigned_tasks(date)
                step.capture(result)
                return {"success": True, "unassigned_tasks": result}
            except Exception as e:
                logger.error(f"Unassigned tasks API error: {e}")
                step.capture({"error": str(e)})
                return format_error_response("Failed to fetch unassigned tasks.", str(e), error_type="api_error")

    async def notify_stakeholders(self, assignments: List[Dict[str, Any]], unassigned_tasks: List[Dict[str, Any]]) -> bool:
        """Sends notifications to employees and managers about assignments and unassigned tasks."""
        async with trace_step(
            "notify_stakeholders", step_type="tool_call",
            decision_summary="Send notifications to stakeholders",
            output_fn=lambda r: f"status={r}"
        ) as step:
            try:
                ok1 = await self.notification_adapter.notify_assignments(assignments)
                ok2 = await self.notification_adapter.notify_unassigned_tasks(unassigned_tasks)
                step.capture(ok1 and ok2)
                return ok1 and ok2
            except Exception as e:
                logger.error(f"Notification error: {e}")
                step.capture(False)
                return False

    def generate_assignment_report(self, assignments, utilization_summary, unassigned_tasks) -> str:
        """Creates formatted reports for management review."""
        with trace_step_sync(
            "generate_assignment_report", step_type="format",
            decision_summary="Format assignment report",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                report = self.reporting_adapter.generate_report(assignments, utilization_summary, unassigned_tasks)
                step.capture(report)
                return report
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                step.capture("Partial report generated due to formatting error.")
                return "Partial report generated due to formatting error."

# --- Main Agent ---
class EmployeeWorkAssignmentAgent:
    """Main agent class, composes orchestrator and adapters."""
    def __init__(self):
        self.orchestrator = AssignmentOrchestrator()

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def assign_tasks(self, tasks: List[Dict[str, Any]], employee_roster: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.orchestrator.assign_tasks(tasks, employee_roster)

    async def generate_utilization_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        return await self.orchestrator.generate_utilization_summary(date)

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def output_unassigned_tasks(self, date: Optional[str] = None) -> Dict[str, Any]:
        return await self.orchestrator.output_unassigned_tasks(date)

    async def notify_stakeholders(self, assignments: List[Dict[str, Any]], unassigned_tasks: List[Dict[str, Any]]) -> bool:
        return await self.orchestrator.notify_stakeholders(assignments, unassigned_tasks)

    def generate_assignment_report(self, assignments, utilization_summary, unassigned_tasks) -> str:
        return self.orchestrator.generate_assignment_report(assignments, utilization_summary, unassigned_tasks)

# --- FastAPI App ---
app = FastAPI(
    title="Employee Work Assignment Agent",
    description="Assigns tasks to employees based on attendance, skills, capacity, and business rules.",
    version="1.0.0"
)

# CORS (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = EmployeeWorkAssignmentAgent()

# --- Exception Handlers ---
@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=format_error_response(
            "Input validation error.",
            tips="Check your JSON structure, required fields, and value types.",
            error_type="validation_error"
        )
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            exc.detail,
            tips="Check your request and try again.",
            error_type="http_error"
        )
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=format_error_response(
            "Internal server error.",
            tips="Contact support if the problem persists.",
            error_type="server_error"
        )
    )

# --- API Endpoints ---
@app.post("/assign-tasks", response_model=AssignTasksResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def assign_tasks_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=format_error_response(
                "Malformed JSON in request body.",
                tips="Check for missing commas, quotes, or brackets. Ensure valid JSON.",
                error_type="json_parse_error"
            )
        )
    try:
        req = AssignTasksRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=format_error_response(
                "Input validation error.",
                tips="Check your JSON structure, required fields, and value types.",
                error_type="validation_error"
            )
        )
    try:
        result = await agent.assign_tasks(
            [t.model_dump() for t in req.tasks],
            [e.model_dump() for e in req.employee_roster]
        )
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Assignment error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                "Assignment processing failed.",
                tips="Contact support if the problem persists.",
                error_type="server_error"
            )
        )

@app.post("/utilization-summary", response_model=UtilizationSummaryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def utilization_summary_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=format_error_response(
                "Malformed JSON in request body.",
                tips="Check for missing commas, quotes, or brackets. Ensure valid JSON.",
                error_type="json_parse_error"
            )
        )
    try:
        req = UtilizationSummaryRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=format_error_response(
                "Input validation error.",
                tips="Check your JSON structure, required fields, and value types.",
                error_type="validation_error"
            )
        )
    try:
        result = await agent.generate_utilization_summary(req.date)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Utilization summary error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                "Utilization summary processing failed.",
                tips="Contact support if the problem persists.",
                error_type="server_error"
            )
        )

@app.post("/unassigned-tasks", response_model=UnassignedTasksResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def unassigned_tasks_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=format_error_response(
                "Malformed JSON in request body.",
                tips="Check for missing commas, quotes, or brackets. Ensure valid JSON.",
                error_type="json_parse_error"
            )
        )
    try:
        req = UnassignedTasksRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=format_error_response(
                "Input validation error.",
                tips="Check your JSON structure, required fields, and value types.",
                error_type="validation_error"
            )
        )
    try:
        result = await agent.output_unassigned_tasks(req.date)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Unassigned tasks error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                "Unassigned tasks processing failed.",
                tips="Contact support if the problem persists.",
                error_type="server_error"
            )
        )

# --- Main Entrypoint ---


async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Employee Work Assignment Agent on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())