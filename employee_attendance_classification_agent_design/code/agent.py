import time as _time
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
import logging
import asyncio
import re
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timedelta
from functools import lru_cache

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator, constr
from dotenv import load_dotenv

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by the runtime.

# Load environment variables from .env if present
load_dotenv()

# Logging configuration
logger = logging.getLogger("attendance_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Azure OpenAI LLM Integration
try:
    import openai
except ImportError:
    openai = None

# Constants for LLM
AZURE_OPENAI_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_ENV = "AZURE_OPENAI_DEPLOYMENT"  # For Azure, deployment name is required

# --- Configuration Management ---

class Config:
    """
    Configuration loader for API keys and LLM settings.
    """
    @staticmethod
    def get_azure_openai_api_key() -> Optional[str]:
        return os.getenv(AZURE_OPENAI_API_KEY_ENV)

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv(AZURE_OPENAI_ENDPOINT_ENV)

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv(AZURE_OPENAI_DEPLOYMENT_ENV)

    @staticmethod
    @trace_agent(agent_name='Employee Attendance Classification Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_llm_config() -> None:
        """
        Validates that all required Azure OpenAI environment variables are set.
        Raises ValueError if any are missing.
        """
        missing = []
        if not Config.get_azure_openai_api_key():
            missing.append(AZURE_OPENAI_API_KEY_ENV)
        if not Config.get_azure_openai_endpoint():
            missing.append(AZURE_OPENAI_ENDPOINT_ENV)
        if not Config.get_azure_openai_deployment():
            missing.append(AZURE_OPENAI_DEPLOYMENT_ENV)
        if missing:
            raise ValueError(f"Missing Azure OpenAI configuration: {', '.join(missing)}")

# --- Utility Functions ---

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(text: str) -> str:
    """
    Masks email addresses and phone numbers in the given text.
    """
    if not text:
        return text
    # Mask emails
    text = re.sub(r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', r'***@***', text)
    # Mask phone numbers (simple pattern)
    text = re.sub(r'\b\d{10,13}\b', '**********', text)
    return text

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    """
    Sanitizes input text by stripping whitespace and removing control characters.
    """
    if not text:
        return text
    text = text.strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text

def is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except Exception:
        return False

def parse_time(time_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(time_str, "%H:%M")
    except Exception:
        return None

def parse_datetime(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except Exception:
        return None

# --- Pydantic Models ---

class CheckInLog(BaseModel):
    time: constr(strip_whitespace=True, min_length=4, max_length=5)  # e.g., "08:55"
    @field_validator("time")
    @classmethod
    def validate_time(cls, v):
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("Check-in time must be in HH:MM format")
        return v

class LeaveData(BaseModel):
    leave_type: constr(strip_whitespace=True, min_length=1)
    approved: bool = True

class ShiftRule(BaseModel):
    shift_start: constr(strip_whitespace=True, min_length=4, max_length=5)  # "09:00"
    shift_end: constr(strip_whitespace=True, min_length=4, max_length=5)    # "18:00"
    start_tolerance_minutes: int = 10
    half_day_threshold_minutes: int = 120

    @field_validator("shift_start", "shift_end")
    @classmethod
    def validate_time(cls, v):
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("Shift time must be in HH:MM format")
        return v

    @field_validator("start_tolerance_minutes", "half_day_threshold_minutes")
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Minutes must be non-negative")
        return v

class HolidayCalendar(BaseModel):
    holidays: List[constr(strip_whitespace=True, min_length=10, max_length=10)]  # "YYYY-MM-DD"

    @field_validator("holidays")
    @classmethod
    def validate_dates(cls, v):
        for date_str in v:
            if not is_valid_date(date_str):
                raise ValueError(f"Invalid holiday date: {date_str}")
        return v

class AttendanceInputData(BaseModel):
    employee_id: constr(strip_whitespace=True, min_length=1)
    date: constr(strip_whitespace=True, min_length=10, max_length=10)  # "YYYY-MM-DD"
    check_in_logs: List[CheckInLog]
    leave_data: Optional[LeaveData] = None
    shift_rules: ShiftRule
    holiday_calendar: HolidayCalendar

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if not is_valid_date(v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def validate_check_in_logs(self):
        if not self.check_in_logs or not isinstance(self.check_in_logs, list):
            raise ValueError("At least one check-in log is required")
        return self

class ProcessedInputData(BaseModel):
    employee_id: str
    date: str
    check_in_time: Optional[str] = None
    leave_status: bool = False
    holiday_status: bool = False
    shift_rules: ShiftRule

class AttendanceStatus(str):
    pass  # Enum-like, but for flexibility

class AuditLogEntry(BaseModel):
    employee_id: str
    date: str
    attendance_status: str
    reason: str
    timestamp: datetime

class NotificationResult(BaseModel):
    employee_id: str
    attendance_status: str
    notification_type: str
    dispatched: bool
    message: str

class ReviewFlagResult(BaseModel):
    employee_id: str
    date: str
    flagged: bool
    reason: str

# --- Error Codes ---

class ErrorCodes:
    ATTENDANCE_POLICY_VIOLATION = "ATTENDANCE_POLICY_VIOLATION"
    DATA_VALIDATION_ERROR = "DATA_VALIDATION_ERROR"

# --- Layered Architecture Classes ---

# Infrastructure Layer: Caching for holidays and shift rules
@lru_cache(maxsize=128)
def get_cached_holidays(holidays: tuple) -> set:
    return set(holidays)

@lru_cache(maxsize=128)
def get_cached_shift_rules(shift_start: str, shift_end: str, start_tol: int, half_day: int) -> ShiftRule:
    return ShiftRule(
        shift_start=shift_start,
        shift_end=shift_end,
        start_tolerance_minutes=start_tol,
        half_day_threshold_minutes=half_day
    )

# --- AttendanceInputProcessor ---

class AttendanceInputProcessor:
    """
    Validates and preprocesses attendance input data.
    """
    def __init__(self):
        pass

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_inputs(self, data: Dict[str, Any]) -> ProcessedInputData:
        """
        Validates and preprocesses raw input data.
        """
        async with trace_step(
            "process_inputs", step_type="parse",
            decision_summary="Validate and preprocess attendance input data",
            output_fn=lambda r: f"employee_id={r.employee_id}, date={r.date}"
        ) as step:
            try:
                validated = AttendanceInputData(**data)
            except ValidationError as ve:
                logger.warning(f"Input validation failed: {ve}")
                raise ValueError(ErrorCodes.DATA_VALIDATION_ERROR)
            # Check for holiday
            holiday_status = validated.date in get_cached_holidays(tuple(validated.holiday_calendar.holidays))
            # Check for leave
            leave_status = bool(validated.leave_data and validated.leave_data.approved)
            # Get earliest check-in time
            check_in_time = None
            if validated.check_in_logs:
                check_in_times = [log.time for log in validated.check_in_logs]
                check_in_time = min(check_in_times)
            processed = ProcessedInputData(
                employee_id=validated.employee_id,
                date=validated.date,
                check_in_time=check_in_time,
                leave_status=leave_status,
                holiday_status=holiday_status,
                shift_rules=validated.shift_rules
            )
            step.capture(processed)
            return processed

    async def validate_inputs(self, data: Dict[str, Any]) -> bool:
        """
        Validates input data for completeness and correctness.
        """
        async with trace_step(
            "validate_inputs", step_type="parse",
            decision_summary="Validate attendance input data for completeness",
            output_fn=lambda r: f"valid={r}"
        ) as step:
            try:
                AttendanceInputData(**data)
                step.capture(True)
                return True
            except ValidationError as ve:
                logger.warning(f"Input validation error: {ve}")
                step.capture(False)
                return False

# --- AttendancePolicyValidator ---

class AttendancePolicyValidator:
    """
    Validates attendance inputs and applies classification policy.
    """
    def __init__(self):
        pass

    async def validate_policy(self, inputs: ProcessedInputData) -> bool:
        """
        Validates processed inputs against attendance policy.
        """
        async with trace_step(
            "validate_policy", step_type="process",
            decision_summary="Validate processed inputs against attendance policy",
            output_fn=lambda r: f"policy_valid={r}"
        ) as step:
            # Policy: Must have shift rules, date, and employee_id
            if not inputs.employee_id or not inputs.date or not inputs.shift_rules:
                logger.warning("Policy validation failed: missing required fields")
                step.capture(False)
                raise ValueError(ErrorCodes.ATTENDANCE_POLICY_VIOLATION)
            step.capture(True)
            return True

# --- AttendanceAuditLogger ---

class AttendanceAuditLogger:
    """
    Logs attendance classification decisions for compliance and audit.
    """
    def __init__(self):
        pass

    async def log_decision(self, employee_id: str, date: str, attendance_status: str, reason: str) -> AuditLogEntry:
        """
        Logs the classification decision.
        """
        async with trace_step(
            "log_decision", step_type="process",
            decision_summary="Log attendance classification decision for audit",
            output_fn=lambda r: f"status={r.attendance_status}"
        ) as step:
            try:
                entry = AuditLogEntry(
                    employee_id=mask_pii(employee_id),
                    date=date,
                    attendance_status=attendance_status,
                    reason=mask_pii(reason),
                    timestamp=datetime.utcnow()
                )
                logger.info(f"Audit log: {entry}")
                step.capture(entry)
                return entry
            except Exception as e:
                logger.error(f"Audit logging failed: {e}")
                raise

# --- NotificationDispatcher ---

class NotificationDispatcher:
    """
    Sends attendance notifications and escalates unresolved cases.
    """
    def __init__(self):
        pass

    async def dispatch_notification(self, employee_id: str, attendance_status: str, notification_type: str) -> NotificationResult:
        """
        Dispatches notification (simulated).
        """
        async with trace_step(
            "dispatch_notification", step_type="process",
            decision_summary="Dispatch attendance notification",
            output_fn=lambda r: f"dispatched={r.dispatched}"
        ) as step:
            try:
                # Simulate notification dispatch
                message = f"Attendance status for {mask_pii(employee_id)} on {notification_type}: {attendance_status}"
                logger.info(f"Notification dispatched: {message}")
                result = NotificationResult(
                    employee_id=mask_pii(employee_id),
                    attendance_status=attendance_status,
                    notification_type=notification_type,
                    dispatched=True,
                    message=message
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Notification dispatch failed: {e}")
                result = NotificationResult(
                    employee_id=mask_pii(employee_id),
                    attendance_status=attendance_status,
                    notification_type=notification_type,
                    dispatched=False,
                    message=str(e)
                )
                step.capture(result)
                return result

# --- ManualReviewSupport ---

class ManualReviewSupport:
    """
    Flags ambiguous or failed attendance records for HR review.
    """
    def __init__(self):
        pass

    async def flag_for_review(self, employee_id: str, date: str, reason: str) -> ReviewFlagResult:
        """
        Flags a record for manual HR review.
        """
        async with trace_step(
            "flag_for_review", step_type="process",
            decision_summary="Flag attendance record for manual HR review",
            output_fn=lambda r: f"flagged={r.flagged}"
        ) as step:
            try:
                logger.info(f"Flagged for review: {employee_id} on {date} - {reason}")
                result = ReviewFlagResult(
                    employee_id=mask_pii(employee_id),
                    date=date,
                    flagged=True,
                    reason=mask_pii(reason)
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Manual review flagging failed: {e}")
                result = ReviewFlagResult(
                    employee_id=mask_pii(employee_id),
                    date=date,
                    flagged=False,
                    reason=str(e)
                )
                step.capture(result)
                return result

# --- AttendanceClassifier (Domain Layer) ---

class AttendanceClassifier:
    """
    Applies strict policy order and business rules to classify attendance status.
    """
    def __init__(self, policy_validator: AttendancePolicyValidator):
        self.policy_validator = policy_validator

    async def classify_attendance(self, employee_id: str, date: str, inputs: ProcessedInputData) -> AttendanceStatus:
        """
        Applies business rules to classify attendance.
        """
        async with trace_step(
            "classify_attendance", step_type="plan",
            decision_summary="Apply strict policy order for attendance classification",
            output_fn=lambda r: f"attendance_status={r}"
        ) as step:
            # Validate policy
            await self.policy_validator.validate_policy(inputs)
            # Strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent
            if inputs.holiday_status:
                status = "Holiday"
            elif inputs.leave_status:
                status = "Leave"
            elif inputs.check_in_time:
                shift_start = parse_time(inputs.shift_rules.shift_start)
                check_in = parse_time(inputs.check_in_time)
                if not shift_start or not check_in:
                    logger.warning("Invalid shift or check-in time")
                    raise ValueError(ErrorCodes.DATA_VALIDATION_ERROR)
                delta = (check_in - shift_start).total_seconds() / 60
                if delta <= inputs.shift_rules.start_tolerance_minutes:
                    status = "Present"
                elif inputs.shift_rules.start_tolerance_minutes < delta <= inputs.shift_rules.half_day_threshold_minutes:
                    status = "Late Present"
                elif delta > inputs.shift_rules.half_day_threshold_minutes:
                    status = "Half Day"
                else:
                    status = "Absent"
            else:
                status = "Absent"
            step.capture(status)
            return status

# --- LLM Interaction Manager (Integration Layer) ---

class LLMInteractionManager:
    """
    Handles LLM calls to Azure OpenAI for attendance classification and communication.
    """
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str, user_prompt_template: str, few_shot_examples: List[str]):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.few_shot_examples = few_shot_examples

    def _get_azure_openai_client(self):
        """
        Lazily initializes and returns the Azure OpenAI client.
        """
        Config.validate_llm_config()
        if openai is None:
            raise ImportError("openai package is not installed")
        api_key = Config.get_azure_openai_api_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )
        return client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def classify_with_llm(self, processed: ProcessedInputData) -> str:
        """
        Calls the LLM to classify attendance and generate a formal message.
        """
        async with trace_step(
            "llm_classification", step_type="llm_call",
            decision_summary="Call Azure OpenAI LLM for attendance classification",
            output_fn=lambda r: f"llm_response={r[:100] if r else ''}"
        ) as step:
            client = self._get_azure_openai_client()
            user_prompt = self._build_user_prompt(processed)
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            for ex in self.few_shot_examples:
                messages.append({"role": "user", "content": ex})
            messages.append({"role": "user", "content": user_prompt})
            _t0 = datetime.utcnow().timestamp()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content if response.choices else ""
            try:
                trace_model_call(
                    provider="azure",
                    model_name=self.model,
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(response.usage, "completion_tokens", 0),
                    latency_ms=int((datetime.utcnow().timestamp() - _t0) * 1000),
                    response_summary=content[:200] if content else ""
                )
            except Exception:
                pass
            step.capture(content)
            return content

    def _build_user_prompt(self, processed: ProcessedInputData) -> str:
        """
        Builds the user prompt for the LLM.
        """
        prompt = (
            f"Employee ID: {processed.employee_id}\n"
            f"Date: {processed.date}\n"
            f"Check-in Time: {processed.check_in_time or 'N/A'}\n"
            f"Leave Status: {'Approved' if processed.leave_status else 'None'}\n"
            f"Holiday Status: {'Yes' if processed.holiday_status else 'No'}\n"
            f"Shift Start: {processed.shift_rules.shift_start}\n"
            f"Shift End: {processed.shift_rules.shift_end}\n"
            f"Shift Start Tolerance (min): {processed.shift_rules.start_tolerance_minutes}\n"
            f"Half Day Threshold (min): {processed.shift_rules.half_day_threshold_minutes}\n"
            f"Please classify the attendance status according to company policy."
        )
        return prompt

# --- Main Agent ---

class AttendanceClassificationAgent:
    """
    Main agent orchestrating attendance classification.
    """
    def __init__(self):
        self.input_processor = AttendanceInputProcessor()
        self.policy_validator = AttendancePolicyValidator()
        self.classifier = AttendanceClassifier(self.policy_validator)
        self.audit_logger = AttendanceAuditLogger()
        self.notifier = NotificationDispatcher()
        self.manual_review = ManualReviewSupport()
        self.llm_manager = LLMInteractionManager(
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a professional attendance classification agent. Your task is to record and classify daily employee attendance using check-in logs, leave data, shift rules, and holiday calendars. Apply strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent. Validate all input data, log every decision, and communicate clearly and formally.",
            user_prompt_template="Please provide the employee's check-in logs, leave data, shift rules, and holiday calendar for attendance classification.",
            few_shot_examples=[
                "Employee A checked in at 08:55, shift starts at 09:00, no leave, not a holiday. Employee A is classified as Present.",
                "Employee B did not check in, has approved leave for the day. Employee B is classified as Leave.",
                "Employee C checked in at 10:30, shift starts at 09:00, no leave, not a holiday. Employee C is classified as Half Day."
            ]
        )

    async def classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for attendance classification.
        """
        async with trace_step(
            "agent_classify", step_type="final",
            decision_summary="End-to-end attendance classification and notification",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            try:
                # Step 1: Validate and preprocess input
                processed = await self.input_processor.process_inputs(data)
                # Step 2: Classify attendance (business rules)
                attendance_status = await self.classifier.classify_attendance(
                    processed.employee_id, processed.date, processed
                )
                # Step 3: LLM confirmation and formal message
                llm_message = await self.llm_manager.classify_with_llm(processed)
                # Step 4: Audit log
                await self.audit_logger.log_decision(
                    processed.employee_id, processed.date, attendance_status, llm_message
                )
                # Step 5: Notification dispatch
                await self.notifier.dispatch_notification(
                    processed.employee_id, attendance_status, "email"
                )
                result = {
                    "success": True,
                    "attendance_status": attendance_status,
                    "llm_message": llm_message,
                    "employee_id": mask_pii(processed.employee_id),
                    "date": processed.date
                }
                step.capture(result)
                return result
            except ValueError as ve:
                logger.warning(f"Classification error: {ve}")
                await self.manual_review.flag_for_review(
                    data.get("employee_id", "unknown"),
                    data.get("date", "unknown"),
                    str(ve)
                )
                result = {
                    "success": False,
                    "error_type": "ValidationError",
                    "error_code": str(ve),
                    "message": "Input validation failed. Please check your data and try again.",
                    "tips": "Ensure all required fields are present and correctly formatted."
                }
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Unhandled error: {e}")
                await self.manual_review.flag_for_review(
                    data.get("employee_id", "unknown"),
                    data.get("date", "unknown"),
                    str(e)
                )
                result = {
                    "success": False,
                    "error_type": "InternalError",
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please contact support.",
                    "tips": "Try again later or contact your HR administrator."
                }
                step.capture(result)
                return result

# --- FastAPI App (Presentation Layer) ---

app = FastAPI(
    title="Employee Attendance Classification Agent",
    description="API for classifying employee attendance using strict HR policy and LLM confirmation.",
    version="1.0.0"
)

agent = AttendanceClassificationAgent()

class AttendanceRequest(BaseModel):
    employee_id: str = Field(..., description="Employee unique identifier")
    date: str = Field(..., description="Attendance date in YYYY-MM-DD format")
    check_in_logs: List[Dict[str, Any]]
    leave_data: Optional[Dict[str, Any]] = None
    shift_rules: Dict[str, Any]
    holiday_calendar: Dict[str, Any]

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        v = sanitize_text(v)
        if not v:
            raise ValueError("Employee ID cannot be empty")
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if not is_valid_date(v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @field_validator("check_in_logs")
    @classmethod
    def validate_check_in_logs(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("At least one check-in log is required")
        return v

    @field_validator("shift_rules")
    @classmethod
    def validate_shift_rules(cls, v):
        if not v or not isinstance(v, dict):
            raise ValueError("Shift rules are required")
        return v

    @field_validator("holiday_calendar")
    @classmethod
    def validate_holiday_calendar(cls, v):
        if not v or not isinstance(v, dict):
            raise ValueError("Holiday calendar is required")
        return v

    @model_validator(mode="after")
    def validate_content_length(self):
        # Limit total input size
        total_chars = (
            len(str(self.employee_id)) +
            len(str(self.date)) +
            sum(len(str(log)) for log in self.check_in_logs) +
            len(str(self.leave_data) or "") +
            len(str(self.shift_rules)) +
            len(str(self.holiday_calendar))
        )
        if total_chars > 50000:
            raise ValueError("Input too large (max 50,000 characters)")
        return self

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Malformed JSON or validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "RequestValidationError",
            "message": "Malformed or invalid JSON in request.",
            "tips": "Check for missing fields, extra commas, or incorrect quotes. Ensure all required fields are present and properly formatted.",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalError",
            "message": "An unexpected error occurred. Please check your input or try again later.",
            "tips": "If the problem persists, contact your HR administrator."
        }
    )

@app.post("/classify", response_model=Dict[str, Any])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def classify_attendance(request: AttendanceRequest):
    """
    Endpoint to classify employee attendance.
    """
    async with trace_step(
        "api_classify_attendance", step_type="final",
        decision_summary="API endpoint for attendance classification",
        output_fn=lambda r: f"success={r.get('success', False)}"
    ) as step:
        try:
            data = request.model_dump()
            _obs_t0 = _time.time()
            result = await agent.classify(data)
            try:
                trace_tool_call(
                    tool_name='agent.classify',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(result)[:200] if result is not None else None,
                    status="success",
                )
            except Exception:
                pass
            step.capture(result)
            return result
        except ValidationError as ve:
            logger.warning(f"Input validation error: {ve}")
            result = {
                "success": False,
                "error_type": "ValidationError",
                "message": "Input validation failed.",
                "tips": "Check your input fields and formatting.",
                "details": str(ve)
            }
            step.capture(result)
            return JSONResponse(status_code=400, content=result)
        except Exception as e:
            logger.error(f"API error: {e}")
            result = {
                "success": False,
                "error_type": "InternalError",
                "message": "An unexpected error occurred.",
                "tips": "Try again later or contact support.",
                "details": str(e)
            }
            step.capture(result)
            return JSONResponse(status_code=500, content=result)

# --- Main Execution Block ---



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
        logger.info("Starting Employee Attendance Classification Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=8080, reload=True)
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