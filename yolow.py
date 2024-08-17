import supervision as sv
import cv2
from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

# import model
model = YOLOWorld(model_id="yolo_world/l")

#set video you want to processing
video_path = 'WhatsApp Video 2024-08-13 at 09.31.19.mp4'
result_video = "output_6.mp4"
generator = sv.get_video_frames_generator(video_path)
frame = next(generator)

# set classes
classes = ["falling object"]
model.set_classes(classes) #you can't put this before inferencing

video_info = sv.VideoInfo.from_video_path(video_path)

width, height = video_info.resolution_wh
frame_area = width * height

annotated_image = frame.copy()

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

frame_generator = sv.get_video_frames_generator(video_path)
video_info = sv.VideoInfo.from_video_path(video_path)

width, height = video_info.resolution_wh
frame_area = width * height
frame_area

with sv.VideoSink(target_path=result_video, video_info=video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        results = model.infer(frame, confidence=0.005)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        detections = detections[(detections.area / frame_area) < 0.10]

        annotated_frame = frame.copy()
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        sink.write_frame(annotated_frame)