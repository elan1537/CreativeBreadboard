const SCALE = 0.25;
let points = [];
let drawState = false;
let previousPoint = [];
let startPoint = [];
let nowPoint = [];
let pointCount = 0;

let send = false;

const img = new Image();
img.src = image_path;
console.log(img.src)

const img_tag = document.querySelector("img");
const canvas = document.getElementById("cropLayer");
const context = canvas.getContext("2d");

img.onload = function () {
    let width_size = parseInt(img.width * SCALE);
    let height_size = parseInt(img.height * SCALE);
    img_tag.width = width_size + 2;
    img_tag.height = height_size + 2;
    canvas.width = width_size;
    canvas.height = height_size;
    img_tag.src = img.src;
};

const draw = (start, target) => {
    context.beginPath();
    context.strokeStyle = "#FF0000";
    context.lineWidth = 2;
    context.arc(...start, 10, 0, 2 * Math.PI);
    context.fillStyle = "red";
    context.fill();
    context.moveTo(...start);
    context.lineTo(...target);
    context.stroke();
    context.closePath();
};

const canvasClear = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    drawState = false;
    startPoint = [];
    pointCount = 0;
    previousPoint = [];
    nowPoint = []
}

window.addEventListener("keydown", (event) => {
    event.preventDefault();
    console.log(pointCount)
    const CODE_ESC = 27;
    console.log("keydown")
    if (event.key === "Escape") {
        canvasClear();
    }
    if (event.key === 'Enter') {
        if(pointCount >= 4) {
            const url = "/points";
            const option = {
                method: 'post',
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    'points': points,
                    'scale': SCALE,
                })
            };

            fetch(url, option).then((response) => {
                console.log(response)
                console.log("send!!");
                canvasClear();
                
                window.location = '/result'
            })
        } else {
            console.log("add point!!")
        }
    }
}, false)

window.addEventListener("keypress", (event) => {
    console.log("keyPress")

})

canvas.addEventListener("click", (event) => {
if (event.button == 0) {
    if (drawState) {
        startPoint = [event.offsetX, event.offsetY];
    } else {
        context.clearRect(0, 0, canvas.width, canvas.height);
        points = [];
        previousPoint = 0;
        pointCount = 0;
        drawState = true;
        pointCount++;
        startPoint = [event.offsetX, event.offsetY];
    }

    if (pointCount >= 4) {
        points.push(startPoint);
        draw(nowPoint, points[0]);
        drawState = false;
    }
}
});

canvas.addEventListener("mousedown", (event) => {
if (event.button == 0) {
    if (pointCount <= 4) {
        pointCount++;
        previousPoint = startPoint;
        points.push(previousPoint);
    }
}
});

canvas.addEventListener("mousemove", (event) => {
if (drawState) {
    context.clearRect(0, 0, canvas.width, canvas.height);

    if (previousPoint) {
        if (points.length != 0)
            for (let i = 0; i < points.length - 1; i++)
            draw(points[i], points[i + 1]);

        draw(previousPoint, startPoint);
    }
    nowPoint = [event.offsetX, event.offsetY];
    draw(startPoint, nowPoint);
}
});