<template>
  <div class="image-modify">
    <div style="position: relative; margin-bottom: 100px">
      <img
        id="imageLayer"
        ref="imageLayer"
        :src="img_src"
        @load="onImageLoad"
        style="z-index: 0"
      />
      <canvas
        ref="canvas"
        id="cropLayer"
        v-on:mousemove="onMouseMove"
        v-on:mousedown="onMouseDown"
        v-on:click="onClick"
        style="position: absolute; left: 0; top: 0; z-index: 1"
      ></canvas>
    </div>
    <div id="result"></div>
  </div>
</template>
<style>
canvas {
  border: 1px solid black;
}
</style>
<script>
export default {
  data: function () {
    return {
      SCALE: 0.25,
      points: [],
      drawState: false,
      previousPoint: [],
      startPoint: [],
      nowPoint: [],
      pointCount: 0,
      canvas: null,
      context: null,
      img_tag: null,
      image_path: require("../../../backend/static/uploads/1_LB.jpeg"),
    };
  },
  props: ["img_src"],
  watch: {},
  created: function () {
    console.log(this.img_src);
  },
  mounted() {
    console.log("mounted");
    console.log(this.img_src);
    this.img_tag = this.$refs.imageLayer;
    this.canvas = document.getElementById("cropLayer");
    this.context = this.canvas.getContext("2d");

    window.addEventListener(
      "keydown",
      (event) => {
        event.preventDefault();
        console.log(this.pointCount);
        console.log("keydown");
        if (event.key === "Escape") {
          this.canvasClear();
        }
        if (event.key === "Enter") {
          console.log(this.points);
          if (this.pointCount >= 4) {
            const url = "http://localhost:3000/points";
            const option = {
              method: "post",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                points: this.points,
                scale: this.SCALE,
              }),
            };
            fetch(url, option).then((response) => {
              console.log(response);
              console.log("send!!");
              this.canvasClear();
              window.location = "/result";
            });
          } else {
            console.log("add point!!");
          }
        }
      },
      false
    );
    window.addEventListener("keypress", () => {
      console.log("keyPress");
    });
  },
  methods: {
    draw(start, target) {
      this.context.beginPath();
      this.context.strokeStyle = "#FF0000";
      this.context.lineWidth = 2;
      this.context.arc(...start, 10, 0, 2 * Math.PI);
      this.context.fillStyle = "red";
      this.context.fill();
      this.context.moveTo(...start);
      this.context.lineTo(...target);
      this.context.stroke();
      this.context.closePath();
    },
    canvasClear() {
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.drawState = false;
      this.startPoint = [];
      this.pointCount = 0;
      this.previousPoint = [];
      this.nowPoint = [];
    },
    onMouseMove(event) {
      if (this.drawState) {
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.previousPoint) {
          if (this.points.length != 0)
            for (let i = 0; i < this.points.length - 1; i++)
              this.draw(this.points[i], this.points[i + 1]);
          this.draw(this.previousPoint, this.startPoint);
        }
        this.nowPoint = [event.offsetX, event.offsetY];
        this.draw(this.startPoint, this.nowPoint);
      }
    },
    onMouseDown(event) {
      if (event.button == 0) {
        if (this.pointCount <= 4) {
          this.pointCount++;
          this.previousPoint = this.startPoint;
          this.points.push(this.previousPoint);
        }
      }
    },
    onClick(event) {
      if (event.button == 0) {
        if (this.drawState) {
          this.startPoint = [event.offsetX, event.offsetY];
        } else {
          this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
          this.points = [];
          this.previousPoint = 0;
          this.pointCount = 0;
          this.drawState = true;
          this.pointCount++;
          this.startPoint = [event.offsetX, event.offsetY];
        }
        if (this.pointCount >= 4) {
          this.points.push(this.startPoint);
          this.draw(this.nowPoint, this.points[0]);
          this.drawState = false;
        }
      }
    },
    onImageLoad() {
      let width_size = parseInt(this.img_tag.width * this.SCALE);
      let height_size = parseInt(this.img_tag.height * this.SCALE);
      this.img_tag.width = width_size + 2;
      this.img_tag.height = height_size + 2;
      this.canvas.width = width_size;
      this.canvas.height = height_size;
    },
    refresh() {
      location.reload();
      console.log("refresh");
    },
  },
};
// var image_path = "{{ url_for('static', filename='uploads/' + image_path) }}";
</script>
