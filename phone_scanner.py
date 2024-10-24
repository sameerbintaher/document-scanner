print("Starting script")
import cv2
print("cv2 imported successfully")
import numpy as np
import base64
import io

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Phone Document Scanner</title>
</head>
<body>
    <h1>Phone Document Scanner</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <button id="snap">Snap Photo</button>
    <canvas id="canvas" width="640" height="480"></canvas>
    <div id="output"></div>

    <script>
        var video = document.getElementById('video');
        var canvas = document.getElementById('canvas');
        var snap = document.getElementById('snap');
        var output = document.getElementById('output');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
            });

        snap.addEventListener("click", function() {
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            var image_data_url = canvas.toDataURL('image/jpeg');

            output.innerHTML = 'Sending image to server...';

            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({image: image_data_url})
            })
            .then(response => response.json())
            .then(data => {
                output.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" />`;
            });
        });
    </script>
</body>
</html>
'''

def process_image(image):
    # Convert base64 image to OpenCV format
    nparr = np.frombuffer(base64.b64decode(image.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral, assume it's our document
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

    # Convert processed image back to base64
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/process', methods=['POST'])
def process():
    image = request.json['image']
    processed_image = process_image(image)
    return {'image': processed_image}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
