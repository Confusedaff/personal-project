"""
Deployment verification tests for the fact-check-llm project.
Tests health, authentication, prediction, and analysis endpoints.
"""
import sys
import time
import requests

BASE = "http://127.0.0.1:8000"
results = []

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def run_tests():
    print("=" * 60)
    print("  fact-check-llm Deployment Verification")
    print("=" * 60)

    # ── 1. Health (unauthenticated) ──
    print("\n--- Health Check (no auth) ---")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        test("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        test("status is 'ok'", data.get("status") == "ok")
        test("model field present", "model" in data)
        test("agents list present", "agents" in data and len(data["agents"]) > 0)
        test("knowledge_base stats present", "knowledge_base" in data)
    except Exception as e:
        test("GET /health reachable", False, str(e))

    # ── 2. Health (authenticated) ──
    print("\n--- Health Check (with auth) ---")
    token = None
    try:
        # Register a test user
        reg = requests.post(f"{BASE}/auth/register", json={
            "username": "testdeploy",
            "email": "testdeploy@example.com",
            "password": "testpass123"
        }, timeout=5)
        test("POST /auth/register returns 201", reg.status_code == 201, f"status={reg.status_code}")

        # Login
        login = requests.post(f"{BASE}/auth/login", data={
            "username": "testdeploy@example.com",
            "password": "testpass123"
        }, timeout=5)
        test("POST /auth/login returns 200", login.status_code == 200, f"status={login.status_code}")
        token = login.json().get("access_token")
        test("access_token received", token is not None and len(token) > 10)

        # Authenticated health
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{BASE}/health", headers=headers, timeout=5)
        test("GET /health (auth) returns 200", r.status_code == 200)
        data = r.json()
        test("user field present in auth response", "user" in data)
        test("review_queue present in auth response", "review_queue" in data)
    except Exception as e:
        test("Auth flow", False, str(e))

    # ── 3. /auth/me ──
    print("\n--- User Info ---")
    if token:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.get(f"{BASE}/auth/me", headers=headers, timeout=5)
            test("GET /auth/me returns 200", r.status_code == 200)
            data = r.json()
            test("username matches", data.get("username") == "testdeploy")
            test("email matches", data.get("email") == "testdeploy@example.com")
        except Exception as e:
            test("GET /auth/me", False, str(e))

    # ── 4. /predict ──
    print("\n--- Legacy Predict ---")
    if token:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.post(f"{BASE}/predict", json={
                "title": "Test headline",
                "text": "This is a test article body for prediction testing."
            }, headers=headers, timeout=10)
            test("POST /predict returns 200", r.status_code == 200, f"status={r.status_code}")
            data = r.json()
            test("label field present", data.get("label") in ("fake", "real"))
            test("confidence field present", isinstance(data.get("confidence"), float))
            test("model_used field present", "model" in data.get("model_used", "").lower() or "svm" in data.get("model_used", "").lower() or "Linear" in data.get("model_used", ""))
        except Exception as e:
            test("POST /predict", False, str(e))

    # ── 5. /analyze (short text) ──
    print("\n--- Multi-Agent Analyze (short text) ---")
    if token:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.post(f"{BASE}/analyze", json={
                "title": "Breaking: Major regulation announced",
                "text": "The government announced new regulations today that will affect the technology industry."
            }, headers=headers, timeout=60)
            test("POST /analyze returns 200", r.status_code == 200, f"status={r.status_code}")
            data = r.json()
            test("label field present", data.get("label") in ("fake", "real", "uncertain"))
            test("confidence field present", isinstance(data.get("confidence"), float))
            test("agent_results present", isinstance(data.get("agent_results"), list) and len(data["agent_results"]) > 0)
            test("evidence_trail present", isinstance(data.get("evidence_trail"), list))
            test("needs_human_review present", isinstance(data.get("needs_human_review"), bool))
            test("input_profile present", data.get("input_profile") != "")
            test("agent_weights present", isinstance(data.get("agent_weights"), dict) and len(data["agent_weights"]) > 0)
        except Exception as e:
            test("POST /analyze", False, str(e))

    # ── 6. /analyze (single word) ──
    print("\n--- Multi-Agent Analyze (single word) ---")
    if token:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.post(f"{BASE}/analyze", json={
                "title": "",
                "text": "Bitcoin"
            }, headers=headers, timeout=60)
            test("POST /analyze (single word) returns 200", r.status_code == 200, f"status={r.status_code}")
            data = r.json()
            test("input_profile is MINIMAL", data.get("input_profile") == "MINIMAL")
            test("label present", data.get("label") in ("fake", "real", "uncertain"))
        except Exception as e:
            test("POST /analyze (single word)", False, str(e))

    # ── 7. Dashboard files ──
    print("\n--- Dashboard Serving ---")
    try:
        r = requests.get(f"{BASE}/", timeout=5)
        test("GET / returns dashboard HTML", r.status_code == 200 and "html" in r.headers.get("content-type", "").lower())

        r = requests.get(f"{BASE}/login", timeout=5)
        test("GET /login returns login page", r.status_code == 200 and "Fake News" in r.text)

        r = requests.get(f"{BASE}/dashboard/auth.js", timeout=5)
        test("GET /dashboard/auth.js serves auth.js", r.status_code == 200 and "FndAuth" in r.text)
        test("auth.js has no hardcoded localhost", "127.0.0.1" not in r.text)
    except Exception as e:
        test("Dashboard serving", False, str(e))

    # ── 8. Unauthorized access ──
    print("\n--- Unauthorized Access ---")
    try:
        r = requests.post(f"{BASE}/predict", json={"text": "test"}, timeout=5)
        test("POST /predict without auth returns 401", r.status_code == 401)

        r = requests.post(f"{BASE}/analyze", json={"text": "test"}, timeout=5)
        test("POST /analyze without auth returns 401", r.status_code == 401)
    except Exception as e:
        test("Unauthorized access", False, str(e))

    # ── Summary ──
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    # Wait for server to be ready
    for i in range(15):
        try:
            requests.get(f"{BASE}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        print("Server not reachable at", BASE)
        sys.exit(1)

    success = run_tests()
    sys.exit(0 if success else 1)
