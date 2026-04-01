
import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict
from flask import Flask, jsonify, request

# --- Fixtures ---

@pytest.fixture
def app():
    """
    Provides a Flask app instance with the /assign-tasks endpoint for testing.
    The endpoint logic is mocked for functional testing.
    """
    app = Flask(__name__)

    @app.route('/assign-tasks', methods=['POST'])
    def assign_tasks():
        data = request.get_json()
        # Simulate validation
        if not data or 'tasks' not in data or 'employees' not in data:
            return jsonify({"success": False, "error": "Invalid input"}), 422
        if not data['tasks'] or not data['employees']:
            return jsonify({"success": False, "error": "No tasks or employees"}), 422

        # Simulate assignment logic
        assignments = []
        unassigned_tasks = []
        utilization_summary = {}
        for task in data['tasks']:
            assigned = False
            for emp in data['employees']:
                if set(task['required_skills']).issubset(set(emp['skills'])) and emp['capacity'] > 0:
                    assignments.append({
                        "employee_id": emp['id'],
                        "task_id": task['id']
                    })
                    emp['capacity'] -= 1
                    utilization_summary[emp['id']] = utilization_summary.get(emp['id'], 0) + 1
                    assigned = True
                    break
            if not assigned:
                unassigned_tasks.append(task['id'])
        return jsonify({
            "success": True,
            "assignments": assignments,
            "utilization_summary": utilization_summary,
            "unassigned_tasks": unassigned_tasks
        }), 200

    return app

@pytest.fixture
def client(app):
    """
    Provides a Flask test client for sending requests to the app.
    """
    return app.test_client()

# --- Test Functions ---

def test_assign_tasks_endpoint_success(client):
    """
    Functional test: Validates that the /assign-tasks endpoint successfully assigns tasks to employees
    when provided with valid input. Checks for correct response structure and content.
    """
    valid_payload = {
        "tasks": [
            {"id": "T1", "required_skills": ["python"]},
            {"id": "T2", "required_skills": ["sql"]}
        ],
        "employees": [
            {"id": "E1", "skills": ["python", "sql"], "capacity": 2},
            {"id": "E2", "skills": ["python"], "capacity": 1}
        ]
    }
    response = client.post('/assign-tasks', json=valid_payload)
    assert response.status_code == 200, "Expected HTTP 200 for valid input"
    resp_json = response.get_json()
    assert resp_json['success'] is True, "Expected 'success' to be True"
    assert isinstance(resp_json['assignments'], list) and len(resp_json['assignments']) > 0, \
        "Expected non-empty 'assignments' list"
    assert isinstance(resp_json['utilization_summary'], dict), \
        "Expected 'utilization_summary' to be a dict"
    assert isinstance(resp_json['unassigned_tasks'], list), \
        "Expected 'unassigned_tasks' to be a list"

def test_assign_tasks_endpoint_internal_error(client, monkeypatch):
    """
    Functional test: Simulates an internal server error (500) from the /assign-tasks endpoint.
    """
    # Patch the endpoint to raise an exception
    with patch('flask.Flask.dispatch_request', side_effect=Exception("Internal error")):
        response = client.post('/assign-tasks', json={
            "tasks": [{"id": "T1", "required_skills": ["python"]}],
            "employees": [{"id": "E1", "skills": ["python"], "capacity": 1}]
        })
        # Flask returns 500 on unhandled exceptions
        assert response.status_code == 500, "Expected HTTP 500 for internal error"

def test_assign_tasks_endpoint_validation_error(client):
    """
    Functional test: Validates that the /assign-tasks endpoint returns 422 when given invalid input.
    """
    # Missing 'tasks' key
    invalid_payload = {
        "employees": [{"id": "E1", "skills": ["python"], "capacity": 1}]
    }
    response = client.post('/assign-tasks', json=invalid_payload)
    assert response.status_code == 422, "Expected HTTP 422 for missing 'tasks'"

    # Empty tasks list
    invalid_payload2 = {
        "tasks": [],
        "employees": [{"id": "E1", "skills": ["python"], "capacity": 1}]
    }
    response2 = client.post('/assign-tasks', json=invalid_payload2)
    assert response2.status_code == 422, "Expected HTTP 422 for empty 'tasks'"

    # Missing JSON body
    response3 = client.post('/assign-tasks')
    assert response3.status_code == 422, "Expected HTTP 422 for missing JSON body"
