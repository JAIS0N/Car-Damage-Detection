import os
from ultralytics import YOLO

def main():
    # Initialize YOLOv8 model (use a pre-trained model or initialize a new one)
    model = YOLO('yolov8s.pt')  # Replace with 'yolov8m.pt', 'yolov8l.pt', etc., if needed
    
    # Train the model
    model.train(
        data='data1.yaml',          # Path to the dataset YAML file
        epochs=20,                 # Number of training epochs
        imgsz=640,                 # Image size
        optimizer='SGD',           # Optimizer (e.g., SGD, Adam)
        lr0=0.01,                  # Initial learning rate
        lrf=0.1,                   # Final learning rate
        momentum=0.937,            # Momentum
        weight_decay=0.0005,       # Weight decay
    )
    # Define the directory where you want to save the model
    save_dir = 'models'
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the trained model
    model.save(os.path.join(save_dir, 'best_model.pt'))

if __name__ == "__main__":
    main()
