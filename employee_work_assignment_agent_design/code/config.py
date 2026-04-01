
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    # --- API Base URL ---
    BASE_URL = "https://workforce.example.com"

    # --- API Endpoints ---
    ENDPOINTS = {
        "attendance_status": "/api/attendance/status",
        "employee_skills": "/api/employees/skills",
        "employee_capacity": "/api/employees/capacity",
        "task_priority": "/api/tasks/priority",
        "task_due_date": "/api/tasks/due-date",
        "task_dependencies": "/api/tasks/dependencies",
        "create_assignments": "/api/assignments",
        "utilization_summary": "/api/assignments/utilization-summary",
        "unassigned_tasks": "/api/assignments/unassigned-tasks"
    }

    # --- API Key Management ---
    @staticmethod
    def get_api_token() -> str:
        token = os.getenv("WORKFORCE_API_TOKEN")
        if not token:
            raise ConfigError("Missing WORKFORCE_API_TOKEN environment variable.")
        return token

    @staticmethod
    def get_azure_openai_key() -> str:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ConfigError("Missing AZURE_OPENAI_API_KEY environment variable.")
        return key

    @staticmethod
    def get_azure_openai_endpoint() -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigError("Missing AZURE_OPENAI_ENDPOINT environment variable.")
        return endpoint

    @staticmethod
    def get_azure_openai_deployment() -> str:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ConfigError("Missing AZURE_OPENAI_DEPLOYMENT environment variable.")
        return deployment

    # --- LLM Configuration ---
    LLM_CONFIG = {
        "provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are an Employee Work Assignment Agent. Assign daily work only to employees marked as available according to attendance status. "
            "Use task priority, due date, required skills, dependencies, and individual capacity to create balanced allocations. "
            "Exclude absent or leave employees, reduce capacity for half-day availability, and output assignments, utilization summary, and clearly labeled unassigned tasks with reasons. "
            "Communicate in a professional, concise, and transparent manner."
        ),
        "user_prompt_template": "Please provide the list of tasks and employee roster for today's assignments.",
        "few_shot_examples": [
            "Assign the following tasks: Task A (high, due today, skill X), Task B (medium, due tomorrow, skill Y). Employees: John (present, skill X, full capacity), Jane (half-day, skill Y, half capacity), Mike (absent).\nAssignments:\n- Task A: John\n- Task B: Jane (capacity adjusted for half-day)\nUtilization Summary:\n- John: 100%\n- Jane: 50%\nUnassigned Tasks:\n- None",
            "Tasks: Task C (high, skill Z), Task D (low, skill Y). Employees: Alice (present, skill Y), Bob (leave).\nAssignments:\n- Task D: Alice\nUtilization Summary:\n- Alice: 50%\nUnassigned Tasks:\n- Task C: No available employee with required skill Z"
        ]
    }

    # --- Domain-Specific Settings ---
    DOMAIN = "general"
    AGENT_NAME = "Employee Work Assignment Agent"
    PERSONALITY = "professional"
    MODALITY = "text"

    # --- Validation and Error Handling ---
    @staticmethod
    def validate():
        errors = []
        try:
            Config.get_api_token()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_key()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_endpoint()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_deployment()
        except ConfigError as e:
            errors.append(str(e))
        if errors:
            raise ConfigError("Configuration validation failed: " + "; ".join(errors))

    # --- Default Values and Fallbacks ---
    @staticmethod
    def get_llm_config() -> Dict[str, Any]:
        return Config.LLM_CONFIG.copy()

    @staticmethod
    def get_endpoint(name: str) -> str:
        return Config.ENDPOINTS.get(name, "")

    @staticmethod
    def get_full_url(endpoint_name: str) -> str:
        endpoint = Config.get_endpoint(endpoint_name)
        if not endpoint:
            raise ConfigError(f"Endpoint '{endpoint_name}' not found in configuration.")
        return f"{Config.BASE_URL}{endpoint}"

    # --- Security Considerations ---
    @staticmethod
    def get_default_headers() -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {Config.get_api_token()}",
            "Content-Type": "application/json"
        }

    # --- Compliance and Logging ---
    AUDIT_LOG_RETENTION_DAYS = 90

# Example usage:
# try:
#     Config.validate()
# except ConfigError as e:
#     print(f"Configuration error: {e}")
