import jetson_inference
import jetson_utils

font = jetson_utils.cudaFont()

camera = jetson_utils.videoSource("/dev/video0")
display = jetson_utils.videoOutput("display://0")
# detector = jetson_inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)
classifier = jetson_inference.imageNet(
    model="/home/nvidia/jetson-inference/python/training/classification/models/productivity2/resnet18.onnx",
    labels="/home/nvidia/jetson-inference/python/training/classification/models/productivity2/labels.txt",
    input_blob="input_0",
    output_blob="output_0"
)

engaged_labels = {
    "working"
}

not_engaged_labels = {
    "sleeping",
    "looking_at_phone"
}
counter = 0
while display.IsStreaming():
    img = camera.Capture()

    class_id, confidence = classifier.Classify(img)

    label = classifier.GetClassDesc(class_id)

    print(f"{label}: {confidence*100:.1f}%")

    if label in engaged_labels:
        attention = "PRODUCTIVE"
        productivity = confidence * 100
    else:
        attention = "NOT PRODUCTIVE"
        productivity = (1 - confidence) * 100

    text = f"{attention} ({label}) {confidence*100:.1f}%"

    font.OverlayText(
        img,
        img.width,
        img.height,
        text,
        50,
        max(0, 100),
        font.White,
        font.Gray40
    )

    summary = f"Productivity: {productivity:.1f}%"

    font.OverlayText(
        img,
        img.width,
        img.height,
        summary,
        10,
        10,
        font.Green,
        font.Gray40
    )

    display.Render(img)
    display.SetStatus(
        f"Classification | {label} ({confidence*100:.1f}%) | {classifier.GetNetworkFPS():.0f} FPS"
    )
        



