
import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- Logging Setup ---
logger = logging.getLogger("attendance_config")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- API Key Management ---
def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(key, default)
    if required and not value:
        logger.error(f"Missing required environment variable: {key}")
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value

# LLM (Azure OpenAI) API keys and endpoints
AZURE_OPENAI_API_KEY = get_env_var("AZURE_OPENAI_API_KEY", required=True)
AZURE_OPENAI_ENDPOINT = get_env_var("AZURE_OPENAI_ENDPOINT", required=True)
AZURE_OPENAI_DEPLOYMENT = get_env_var("AZURE_OPENAI_DEPLOYMENT", required=True)

# --- LLM Configuration ---
LLM_CONFIG: Dict[str, Any] = {
    "provider": "azure",
    "model": "gpt-4.1",
    "temperature": 0.7,
    "max_tokens": 2000,
    "system_prompt": (
        "You are a professional attendance classification agent. Your task is to record and classify daily employee attendance using check-in logs, leave data, shift rules, and holiday calendars. "
        "Apply strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent. Validate all input data, log every decision, and communicate clearly and formally."
    ),
    "user_prompt_template": "Please provide the employee's check-in logs, leave data, shift rules, and holiday calendar for attendance classification.",
    "few_shot_examples": [
        "Employee A checked in at 08:55, shift starts at 09:00, no leave, not a holiday. Employee A is classified as Present.",
        "Employee B did not check in, has approved leave for the day. Employee B is classified as Leave.",
        "Employee C checked in at 10:30, shift starts at 09:00, no leave, not a holiday. Employee C is classified as Half Day."
    ]
}

# --- Domain-Specific Settings ---
DOMAIN = "general"
AGENT_NAME = "Employee Attendance Classification Agent"
POLICY_ORDER = [
    "Holiday",
    "Leave",
    "Present",
    "Late Present",
    "Half Day",
    "Absent"
]
REQUIRED_CONFIG_KEYS = [
    "shift_rules",
    "holiday_calendar",
    "leave_data",
    "check_in_logs"
]

# --- Validation and Error Handling ---
ERROR_CODES = {
    "ATTENDANCE_POLICY_VIOLATION": "Attendance policy violation.",
    "DATA_VALIDATION_ERROR": "Input data validation error."
}

def validate_llm_config(config: Dict[str, Any]) -> None:
    required_keys = ["provider", "model", "system_prompt", "user_prompt_template"]
    for key in required_keys:
        if key not in config or not config[key]:
            logger.error(f"Missing LLM configuration key: {key}")
            raise ValueError(f"Missing LLM configuration key: {key}")

def validate_api_keys() -> None:
    if not AZURE_OPENAI_API_KEY:
        logger.error("Azure OpenAI API key is missing.")
        raise EnvironmentError("Azure OpenAI API key is missing.")
    if not AZURE_OPENAI_ENDPOINT:
        logger.error("Azure OpenAI endpoint is missing.")
        raise EnvironmentError("Azure OpenAI endpoint is missing.")
    if not AZURE_OPENAI_DEPLOYMENT:
        logger.error("Azure OpenAI deployment name is missing.")
        raise EnvironmentError("Azure OpenAI deployment name is missing.")

# --- Default Values and Fallbacks ---
DEFAULT_SHIFT_RULES = {
    "shift_start": "09:00",
    "shift_end": "18:00",
    "start_tolerance_minutes": 10,
    "half_day_threshold_minutes": 120
}
DEFAULT_HOLIDAY_CALENDAR = {
    "holidays": []
}
DEFAULT_LEAVE_DATA = None
DEFAULT_CHECK_IN_LOGS = []

def get_default_config() -> Dict[str, Any]:
    return {
        "shift_rules": DEFAULT_SHIFT_RULES,
        "holiday_calendar": DEFAULT_HOLIDAY_CALENDAR,
        "leave_data": DEFAULT_LEAVE_DATA,
        "check_in_logs": DEFAULT_CHECK_IN_LOGS
    }

# --- API Requirements (Knowledge Base) ---
API_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "name": None,
        "type": None,
        "purpose": None,
        "authentication": None,
        "rate_limits": None,
        "http_method": None,
        "endpoint": None,
        "full_url": None,
        "headers_required": None,
        "query_params": None,
        "request_body": None,
        "response_format": None,
        "source": "knowledge_base"
    }
]

# --- Exported Config Object ---
class AgentConfig:
    agent_name: str = AGENT_NAME
    domain: str = DOMAIN
    llm_config: Dict[str, Any] = LLM_CONFIG
    api_key: str = AZURE_OPENAI_API_KEY
    api_endpoint: str = AZURE_OPENAI_ENDPOINT
    api_deployment: str = AZURE_OPENAI_DEPLOYMENT
    policy_order: List[str] = POLICY_ORDER
    required_config_keys: List[str] = REQUIRED_CONFIG_KEYS
    error_codes: Dict[str, str] = ERROR_CODES
    api_requirements: List[Dict[str, Any]] = API_REQUIREMENTS
    default_values: Dict[str, Any] = get_default_config()

    @classmethod
    def validate(cls):
        validate_llm_config(cls.llm_config)
        validate_api_keys()

# Validate config on import
try:
    AgentConfig.validate()
except Exception as e:
    logger.error(f"Agent configuration validation failed: {e}")
    raise

