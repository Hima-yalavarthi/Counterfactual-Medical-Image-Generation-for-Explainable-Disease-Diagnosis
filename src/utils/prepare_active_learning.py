import pandas as pd
import json
import os

def prepare_refined_dataset(predictions_csv, feedback_json, output_csv):
    """
    Merges clinician feedback with the original predictions to create 
    a refined dataset for future retraining/fine-tuning.
    """
    if not os.path.exists(predictions_csv):
        print(f"Error: {predictions_csv} not found.")
        return
        
    df = pd.read_csv(predictions_csv)
    
    if os.path.exists(feedback_json):
        with open(feedback_json, 'r') as f:
            feedback_data = json.load(f)
            
        print(f"Loaded {len(feedback_data)} clinician validations.")
        
        # Create a mapping of patient_id -> clinician_label
        feedback_map = {entry['patient_id']: entry['clinician_label'] for entry in feedback_data}
        
        # Update labels in the dataframe
        updates = 0
        for idx, row in df.iterrows():
            if row['filename'] in feedback_map:
                new_label = feedback_map[row['filename']]
                if df.at[idx, 'predicted_label'] != new_label:
                    updates += 1
                df.at[idx, 'true_label'] = new_label # Update 'true_label' with clinician validation
        
        print(f"Refined {updates} labels based on clinician feedback.")
    else:
        print("No clinician feedback found. Returning original predictions.")

    # Save the refined dataset
    df.to_csv(output_csv, index=False)
    print(f"Refined dataset saved to {output_csv}")

if __name__ == "__main__":
    prepare_refined_dataset(
        predictions_csv='results/predictions.csv',
        feedback_json='results/clinician_feedback.json',
        output_csv='data/refined_labels.csv'
    )
