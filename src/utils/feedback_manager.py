import json
import os
import datetime

FEEDBACK_FILE = 'results/clinician_feedback.json'

def save_feedback(patient_id, ai_label, clinician_label, notes=""):
    """
    Saves clinician feedback to a persistent JSON file.
    """
    os.makedirs('outputs', exist_ok=True)
    
    feedback_entry = {
        'patient_id': patient_id,
        'ai_label': ai_label,
        'clinician_label': clinician_label,
        'notes': notes,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_discrepancy': ai_label != clinician_label
    }
    
    # Load existing feedback
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = []
    else:
        data = []
        
    # Check if this patient already has feedback, if so, update it
    found = False
    for i, entry in enumerate(data):
        if entry['patient_id'] == patient_id:
            data[i] = feedback_entry
            found = True
            break
            
    if not found:
        data.append(feedback_entry)
        
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    return True

def get_feedback_summary():
    """
    Returns a summary of clinician feedback.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {'total': 0, 'discrepancies': 0}
        
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            data = json.load(f)
    except:
        return {'total': 0, 'discrepancies': 0}
        
    total = len(data)
    discrepancies = sum(1 for entry in data if entry['is_discrepancy'])
    
    return {
        'total': total,
        'discrepancies': discrepancies,
        'feedback_data': data
    }
