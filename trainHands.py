from ultralytics import YOLO

def train_model():
    
    model = YOLO('yolov8n.pt')  

    model.train(
        data='data.yaml',  
        epochs=100,        
        imgsz=640,          
        batch=16,           
        workers=2,         
        project='runs/train', 
        name='yolov8n_hands_detection'  
    )

if __name__ == '__main__':
    train_model()
