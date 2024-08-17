from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw
import torch
import cv2
import numpy as np

# Load the model and processor
model_id = 'microsoft/Florence-2-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
model = model.to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Define a function to run inference
def run_inference(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

# Define a function to plot bounding boxes
def plot_bbox(image, data):
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(data['bboxes'], data['labels']):  
        if "falling" in label.lower():  # Filter for labels containing "falling"
            x1, y1, x2, y2 = bbox  
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), label, fill="red")
    return image

# Load the video
video_path = "videoplayback.mp4"  # Replace with your video path
output_path = "data_3.mp4"  # Output video path

cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection with caption-to-phrase grounding
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = run_inference(task_prompt, image, text_input="falling object")
    print(results)

    # Annotate the frame with bounding boxes
    annotated_image = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    
    # Convert the PIL image back to an OpenCV frame
    frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
