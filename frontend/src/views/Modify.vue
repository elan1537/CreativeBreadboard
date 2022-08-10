<template>
  <div class="container">
    <h1>Modify View</h1>
    <div class="row">
      <div class="col-md-7 ms-auto">
        <div style="position: relative">
          <img
            id="imageLayer"
            ref="imageLayer"
            :src="uploaded_img"
            style="z-index: 0"
            @load="onImageLoad"
          />
          <canvas
            id="cropLayer"
            ref="canvas"
            style="position: absolute; left: 0; top: 0; z-index: 1"
            @mousemove="onMove"
            @mousedown="onDown"
          />
        </div>
      </div>
      <div class="col-md-5">
        <div class="row gx-4 gx-lg-5">
          <div class="col">
            <CardBody :title="title_1" :text="'저항영역을 수정해주세요'">
              <template #footer>
                <div class="row">
                  <button
                    type="button"
                    class="btn btn-primary"
                    data-bs-toggle="modal"
                    data-bs-target="#exampleModal2"
                  >
                    modify
                  </button>
                  <div
                    id="exampleModal2"
                    class="modal fade"
                    tabindex="-1"
                    aria-labelledby="exampleModalLabel2"
                    aria-hidden="true"
                  >
                    <div class="modal-dialog">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h2 id="exampleModalLabel2" class="modal-title">
                            저항영역을 수정하세요
                          </h2>
                          <button
                            type="button"
                            class="btn-close"
                            data-bs-dismiss="modal"
                            aria-label="Close"
                          />
                        </div>
                        <div class="modal-body">
                          <div
                            v-for="(row, idx) in temp_area_points"
                            :key="`${row}_${idx}`"
                            class="row mb-3"
                          >
                            <div class="col">
                              {{ row }}
                            </div>
                          </div>
                        </div>
                        <div class="modal-footer">
                          <button
                            type="button"
                            class="btn btn-secondary"
                            data-bs-dismiss="modal"
                            aria-label="Close"
                          >
                            Close
                          </button>
                          <button
                            type="button"
                            class="btn btn-primary"
                            data-bs-dismiss="modal"
                            @click="setResistorArea"
                          >
                            Save changes
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </CardBody>
          </div>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="col-md-7 ms-auto">
        <img
          class="img-fluid rounded mb-4 mb-lg-0"
          :src="circuit_img"
          width="600"
          height="800"
          alt="..."
        />
      </div>
      <div class="col-md-5">
        <div class="row gx-4 gx-lg-5">
          <div class="col">
            <CardBody :title="title_2" :text="'저항값을 수정해주세요'">
              <template #footer>
                <div class="row">
                  <button
                    type="button"
                    class="btn btn-primary"
                    data-bs-toggle="modal"
                    data-bs-target="#exampleModal"
                  >
                    modify
                  </button>
                  <!-- Modal -->
                  <div
                    id="exampleModal"
                    class="modal fade"
                    tabindex="-1"
                    aria-labelledby="exampleModalLabel"
                    aria-hidden="true"
                  >
                    <div class="modal-dialog">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h2 id="exampleModalLabel" class="modal-title">
                            저항값을 입력하세요
                          </h2>
                          <button
                            type="button"
                            class="btn-close"
                            aria-label="Close"
                          />
                        </div>
                        <div class="modal-body">
                          <div
                            v-for="(row, idx) in circuit"
                            :key="`${row}_${idx}`"
                            class="row mb-3"
                          >
                            <label
                              :for="row['name']"
                              class="col-sm-3 col-form-label"
                              >{{ row["name"] }}</label
                            >
                            <div class="col">
                              <input
                                :id="row['name']"
                                type="number"
                                class="form-control"
                                :placeholder="row['value']"
                                :value="row['value']"
                                @input="setResistorValue($event, row['name'])"
                              />
                            </div>
                          </div>
                        </div>
                        <div class="modal-footer">
                          <button
                            type="button"
                            class="btn btn-secondary"
                            data-bs-dismiss="modal"
                            aria-label="Close"
                          >
                            Close
                          </button>
                          <button
                            type="button"
                            class="btn btn-primary"
                            data-bs-dismiss="modal"
                            @click="onSaveButton"
                          >
                            Save changes
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </CardBody>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import CardBody from "../components/CardBody.vue";
import axios from "axios";

export default {
  name: "ModifyView",
  components: { CardBody },
  data() {
    return {
      circuit: [
        [{ name: "Resistor", value: 1500 }],
        [
          { name: "Resistor", value: 2000 },
          { name: "Resistor", value: 340 },
        ],
        [{ name: "Resistor", value: 500 }],
      ],
      uploaded_img: "",
      title_1: "저항영역 수정하기",
      title_2: "저항값 입력하기",
      scale: null,
      canvas: null,
      context: null,
      img_tag: null,
      area_points: null,
      temp_area_points: {},
      circuit_img: null,
      isDrawing: false,
      isDraging: false,
      startX: null,
      startY: null,
      cbRGB: null,
      finalWidth: null,
      finalHeight: null,
      isSetResistorArea: false,
      detected_components: null,
      components: null,
    };
  },
  created() {
    console.log("Created");

    const jsonObj = this.$route.query;
    console.log(jsonObj.components);
    console.log(jsonObj === undefined);
    if (jsonObj !== undefined) {
      if (jsonObj.components !== undefined) {
        localStorage.components = jsonObj.components;
      }
    }

    if (localStorage.circuit_img) {
      axios({
        url: "http://localhost:3000/warpedImg",
        method: "get",
        headers: { "Content-Type": "application/json" },
      }).then((response) => {
        const imgData = response.data.img;
        this.uploaded_img = "data:image/jpg;base64," + imgData;
      });
      this.circuit_img = "data:image/png;base64," + localStorage.circuit_img;
      this.detected_components = JSON.parse(localStorage.detected_components);
      this.area_points = this.detected_components.resistor_body;
    } else {
      axios({
        url: "/draw",
        method: "get",
        headers: { "Content-Type": "multipart/form-data" },
      }).then((response) => {
        const imgData = response.data.circuit;
        localStorage.circuit_img = imgData;
        this.circuit_img = imgData;
      });
    }

    axios({
      url: "http://localhost:3000/resistor",
      method: "get",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((response) => {
      const responseData = response.data.data;
      const tempArr = [];
      for (let i = 0; i < responseData.length; i++) {
        for (let j = 0; j < responseData[i].length; j++) {
          tempArr.push(responseData[i][j]);
        }
      }
      this.circuit = tempArr;
    });
  },
  mounted() {
    window.addEventListener("mouseup", this.stopDrag);
    window.addEventListener("keydown", this.onKeydown);

    this.uploaded_img = this.origin_img;
    this.img_tag = this.$refs.imageLayer;
    this.canvas = this.$refs.canvas;
    this.context = this.canvas.getContext("2d");
  },
  methods: {
    uploadImg() {
      const image = this.$refs.image.files[0];
      const url = URL.createObjectURL(image);
      this.image = url;
    },
    setResistorArea() {
      if (Object.keys(this.temp_area_points).length === 0) return;

      Object.keys(this.temp_area_points).map((key) => {
        const row = this.temp_area_points[key];
        this.area_points[key] = row;
      });

      const temp = JSON.parse(localStorage.detected_components);
      temp.resistor_body = this.area_points;

      localStorage.detected_components = JSON.stringify(temp);

      axios({
        method: "post",
        url: "/area",
        data: JSON.stringify(this.temp_area_points),
        headers: {
          "Content-Type": "application/json",
        },
      }).then(() => {
        axios({
          url: "/draw",
          method: "get",
          headers: { "Content-Type": "multipart/form-data" },
        }).then((response) => {
          const imgData = "data:image/png;base64," + response.data.circuit;
          localStorage.circuit_img = response.data.circuit;
          this.circuit_img = imgData;
          axios({
            url: "/calc",
            method: "get",
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            localStorage.circuit_analysis = JSON.stringify(
              response.data.circuit_analysis
            );
            this.temp_area_points = {};
          });
        });
      });
    },
    setResistorValue(event, name) {
      this.circuit.map((row) => {
        if (row.name === name) {
          row.value = parseInt(event.target.value);
        }
      });
    },
    onSaveButton() {
      axios({
        method: "post",
        url: "http://localhost:3000/resistor",
        headers: {
          "Content-Type": "application/json",
        },
        data: JSON.stringify(this.circuit),
      }).then(() => {
        axios({
          url: "/draw",
          method: "get",
          headers: { "Content-Type": "multipart/form-data" },
        }).then((response) => {
          const imgData = response.data.circuit;

          axios({
            url: "/calc",
            method: "get",
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            console.log(response.data);
            localStorage.circuit_analysis = JSON.stringify(
              response.data.circuit_analysis
            );
          });

          localStorage.circuit_img = imgData;
          this.circuit_img = "data:image/png;base64," + imgData;
          // window.location.reload();
        });
      });
    },
    onImageLoad() {
      console.log("onImageLoad");
      const img = new Image();
      img.src = this.uploaded_img;
      this.scale = localStorage.scale;

      img.onload = () => {
        const widthSize = parseInt(img.width * this.scale);
        const heightSize = parseInt(img.height * this.scale);
        const resistorArea = this.detected_components.resistor_body;

        if (Object.keys(resistorArea).length === 0) return;

        this.img_tag.width = widthSize + 2;
        this.img_tag.height = heightSize + 2;
        this.canvas.width = widthSize;
        this.canvas.height = heightSize;

        const components = JSON.parse(localStorage.components);
        // Resistor
        Object.keys(components.Resistor).forEach((key) => {
          const row = components.Resistor[key][0];
          const startCoord = row.startCoord;
          const endCoord = row.endCoord;
          console.log(startCoord[0] * this.scale, startCoord[1] * this.scale);
          this.context.beginPath();
          this.context.fillStyle = "red";
          this.context.font = "20px Arial";
          this.context.fillText(
            row.start,
            startCoord[0] * this.scale,
            startCoord[1] * this.scale - 20
          );
          this.context.arc(
            startCoord[0] * this.scale,
            startCoord[1] * this.scale,
            5,
            0,
            Math.PI * 2,
            true
          );
          this.context.fill();
          this.context.closePath();
          this.context.beginPath();
          this.context.fillStyle = "blue";
          this.context.arc(
            endCoord[0] * this.scale,
            endCoord[1] * this.scale,
            5,
            0,
            Math.PI * 2,
            true
          );
          this.context.font = "20px Arial";
          this.context.fillStyle = "orange";
          this.context.fillText(
            row.end,
            endCoord[0] * this.scale,
            endCoord[1] * this.scale + 20
          );
          this.context.fill();
          this.context.closePath();
        });

        Object.keys(components.Line).forEach((key) => {
          console.log(key);
        });

        Object.keys(resistorArea).forEach((key) => {
          const row = resistorArea[key];

          const [xmin, ymin, xmax, ymax] = [
            row.xmin * this.scale,
            row.ymin * this.scale,
            row.xmax * this.scale,
            row.ymax * this.scale,
          ];

          this.context.beginPath();
          this.context.lineWidth = 4;
          this.context.strokeStyle = `rgb(${row.cbRGB[0]}, ${row.cbRGB[1]}, ${row.cbRGB[2]})`;
          this.context.rect(xmin, ymin, xmax - xmin, ymax - ymin);
          this.context.stroke();
          this.context.closePath();
        });
      };
    },
    drawingAreas(points) {
      Object.keys(points).map((key) => {
        const row = points[key];
        const xmin = row.xmin * this.scale;
        const ymin = row.ymin * this.scale;
        const xmax = row.xmax * this.scale;
        const ymax = row.ymax * this.scale;

        this.context.beginPath();
        this.context.lineWidth = 4;
        this.context.strokeStyle = `rgb(${row.cbRGB[0]}, ${row.cbRGB[1]}, ${row.cbRGB[2]})`;
        this.context.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        this.context.stroke();
        this.context.closePath();
      });
    },

    onMove(event) {
      if (!this.isDraging) return;

      const currentX = event.offsetX;
      const currentY = event.offsetY;

      this.finalWidth = currentX - this.startX;
      this.finalHeight = currentY - this.startY;

      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.drawingAreas(this.area_points);

      if (this.temp_area_points !== null) {
        this.drawingAreas(this.temp_area_points);
      }

      this.context.beginPath();
      this.context.lineWidth = 4;
      this.context.strokeStyle = `rgb(${this.cbRGB[0]}, ${this.cbRGB[1]}, ${this.cbRGB[2]})`;
      this.context.strokeRect(
        this.startX,
        this.startY,
        this.finalWidth,
        this.finalHeight
      );
      this.context.stroke();
      this.context.closePath();
    },
    onDown(event) {
      console.log(event.offsetX, event.offsetY);

      this.isDraging = true;
      this.startX = event.offsetX;
      this.startY = event.offsetY;
      this.cbRGB = [
        parseInt(Math.random() * 255),
        parseInt(Math.random() * 255),
        parseInt(Math.random() * 255),
      ];
      this.context.beginPath();
      this.context.lineWidth = 4;
      this.strokeStyle = `rgb(${this.cbRGB[0]}, ${this.cbRGB[1]}, ${this.cbRGB[2]})`;
      this.context.rect(
        this.startX,
        this.startY,
        this.finalWidth,
        this.finalHeight
      );
      this.context.stroke();
      this.context.closePath();
    },
    stopDrag() {
      let lastKey;
      Object.keys(this.temp_area_points).length === 0
        ? (lastKey = Object.keys(this.area_points))
        : (lastKey = Object.keys(this.temp_area_points));
      lastKey = parseInt(lastKey[lastKey.length - 1]);
      lastKey += 1;

      let xmin = 0;
      let ymin = 0;
      let xmax = 0;
      let ymax = 0;
      let wid = 0;
      let hei = 0;

      xmin = this.startX;
      ymin = this.startY;
      xmax = this.startX + this.finalWidth;
      ymax = this.startY + this.finalHeight;

      const yDif = ymax - ymin;

      if (yDif < 0) {
        const [t1, t2] = [xmin, ymin];
        xmin = xmax;
        ymin = ymax;
        xmax = t1;
        ymax = t2;
      }

      xmin /= this.scale;
      ymin /= this.scale;
      xmax /= this.scale;
      ymax /= this.scale;

      wid = Math.abs(this.finalWidth) / this.scale;
      hei = Math.abs(this.finalHeight) / this.scale;

      if (
        (this.finalWidth >= 0 && this.finalWidth <= 30) ||
        (this.finalHeight >= 0 && this.finalHeight <= 30)
      ) {
        console.log("No save");
        this.isDraging = false;
        this.isDrawing = false;
        this.startX = null;
        this.startY = null;
        this.finalHeight = null;
        this.finalWidth = null;
      } else {
        const newObj = {
          xmin,
          ymin,
          length: hei,
          width: wid,
          xmax,
          ymax,
          confidence: 1.0,
          cbRGB: this.cbRGB,
        };

        this.temp_area_points[lastKey] = newObj;

        this.isDraging = false;
        this.startX = null;
        this.startY = null;
        this.finalHeight = null;
        this.finalWidth = null;
        this.isDrawing = false;
      }
    },
    canvasClear() {
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    },
    onKeydown(event) {
      if (event.key === "Escape") {
        this.canvasClear();
        this.isDraging = false;
        this.isDrawing = false;
        this.startX = null;
        this.startY = null;
        this.finalHeight = null;
        this.finalWidth = null;
        this.temp_area_points = {};
        Object.keys(this.area_points).map((key) => {
          const row = this.area_points[key];
          const xmin = row.xmin * this.scale;
          const ymin = row.ymin * this.scale;
          const xmax = row.xmax * this.scale;
          const ymax = row.ymax * this.scale;

          this.context.beginPath();
          this.context.lineWidth = 4;
          this.context.strokeStyle = `rgb(${row.cbRGB[0]}, ${row.cbRGB[1]}, ${row.cbRGB[2]})`;
          this.context.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
          this.context.stroke();
          this.context.closePath();
        });
      }
    },
  },
};
</script>
