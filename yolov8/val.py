from ultralytics import YOLO
import json

def main():
    # Load the trained model
    model = YOLO('models/best_model.pt')
    
    # Perform validation
    results = model.val(data='data1.yaml')
    
    # Extract evaluation metrics directly from the results object
    metrics = {
        'precision': results.box.map50,  # mAP at IoU=0.50
        'recall': results.box.map,       # mAP at IoU=0.50:0.95
        'mAP_50': results.box.map50,     # mAP at IoU=0.50
        'mAP_50_95': results.box.map,    # mAP at IoU=0.50:0.95
        'f1_score': results.box.f1,      # F1 score
    }
    
    # Save the metrics to a JSON file
    with open('results/validation_reports.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
