<template>
  <div class="image-modify">
    <div style="position: relative">
      <img
        id="imageLayer"
        ref="imageLayer"
        :src="imgSrc"
        style="z-index: 0"
        @load="onImageLoad"
      />
      <canvas
        id="cropLayer"
        ref="canvas"
        style="position: absolute; left: 0; top: 0; z-index: 1"
        @mousemove="onMouseMove"
        @mousedown="onMouseDown"
        @click="onClick"
      ></canvas>
    </div>
    <div id="result"></div>
  </div>
</template>
<script>
export default {
  name: "ComponentImageModify",
  props: {
    imgSrc: {
      type: String,
      required: true,
    },
    isSuccess: Boolean,
  },
  emits: ["is-success", "point-count", "send-data"],
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
    };
  },
  watch: {
    isSuccess: function () {
      if (this.isSuccess === true) {
        this.canvasClear();
        this.$emit("is-success", false);
      }
    },
  },
  mounted() {
    this.img_tag = this.$refs.imageLayer;
    this.canvas = document.getElementById("cropLayer");
    this.context = this.canvas.getContext("2d");

    window.addEventListener(
      "keydown",
      (event) => {
        event.preventDefault();
        if (event.key === "Escape") {
          this.canvasClear();
          this.pointCount = 0;

          this.$emit("point-count", this.pointCount);
          this.$emit("send-data", {});
        }
      },
      false
    );
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
          if (this.points.length !== 0)
            for (let i = 0; i < this.points.length - 1; i++)
              this.draw(this.points[i], this.points[i + 1]);
          this.draw(this.previousPoint, this.startPoint);
        }
        this.nowPoint = [event.offsetX, event.offsetY];
        this.draw(this.startPoint, this.nowPoint);
      }
    },
    onMouseDown(event) {
      if (event.button === 0) {
        if (this.pointCount <= 4) {
          const data = { points: this.points, scale: this.SCALE };
          this.pointCount++;
          this.$emit("point-count", this.pointCount);
          this.$emit("send-data", data);
          this.previousPoint = this.startPoint;
          this.points.push(this.previousPoint);
        }
      }
    },
    onClick(event) {
      if (event.button === 0) {
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
      const img = new Image();
      img.src = this.imgSrc;
      img.onload = () => {
        const widthSize = parseInt(img.width * this.SCALE);
        const heightSize = parseInt(img.height * this.SCALE);
        this.img_tag.width = widthSize + 2;
        this.img_tag.height = heightSize + 2;
        this.canvas.width = widthSize;
        this.canvas.height = heightSize;
      };
    },
    refresh() {
      location.reload();
      console.log("refresh");
    },
  },
};
// var image_path = "{{ url_for('static', filename='uploads/' + image_path) }}";
</script>
<style>
canvas {
  border: 1px solid black;
}
</style>
