import requests
import json
import time

BASE_URL = "http://localhost:8080/api/v1"

def test_health():
    print("Checking API Health...")
    try:
        response = requests.get(f"{BASE_URL}/patients")
        if response.status_code == 200:
            patients = response.json().get("patients", [])
            print(f"✅ API is up. Found {len(patients)} patients.")
            return patients
        else:
            print(f"❌ API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
    return None

def test_analyze(patient):
    print(f"\nTesting Analysis for Subject: {patient['id']}...")
    payload = {
        "subject_id": patient['id'],
        "matrix_path": patient['matrix_path'],
        "t1_path": patient['t1_path'],
        "fmri_weight": 0.5  # Backend now handles this, but parameter might still be required by FastAPI
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/analyze", data=payload)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis Successful ({elapsed:.2f}s)")
            print(f"   Modality: {'Dual' if patient['t1_path'] else 'fMRI-only'}")
            for task, res in data['task_results'].items():
                print(f"   - {task}: Pred={res['prediction']}, Prob={res['prob_fused']:.4f}")
            return data
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
    return None

def test_report(analyze_data):
    if not analyze_data: return
    print(f"\nTesting Report Generation for Subject: {analyze_data['subject_id']}...")
    
    payload = {
        "subject_id": analyze_data['subject_id'],
        "task_results": analyze_data['task_results'],
        "kg_context": analyze_data['kg_context'],
        "mode": "fast"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/report/stream", json=payload, stream=True)
        if response.status_code == 200:
            print("✅ Report Stream Started. Receiving first 100 chars...")
            content = ""
            for chunk in response.iter_content(chunk_size=100):
                if chunk:
                    content += chunk.decode('utf-8')
                    if len(content) > 100: break
            print(f"   Content Preview: {content[:100]}...")
        else:
            print(f"❌ Report failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Report Error: {e}")

if __name__ == "__main__":
    patients = test_health()
    if patients:
        # Test one fMRI only and one Dual Modal if available
        fmri_only = next((p for p in patients if not p['t1_path']), patients[0])
        dual_modal = next((p for p in patients if p['t1_path']), None)
        
        test_analyze(fmri_only)
        if dual_modal:
            analyze_res = test_analyze(dual_modal)
            test_report(analyze_res)
