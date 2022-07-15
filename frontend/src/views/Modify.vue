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
            @load="onImageLoad"
            style="z-index: 0"
          />
          <canvas
            ref="canvas"
            id="cropLayer"
            style="position: absolute; left: 0; top: 0; z-index: 1"
            @mousemove="onMove"
            @mousedown="onDown"
          ></canvas>
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
                    class="modal fade"
                    id="exampleModal2"
                    tabindex="-1"
                    aria-labelledby="exampleModalLabel2"
                    aria-hidden="true"
                  >
                    <div class="modal-dialog">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h2 class="modal-title" id="exampleModalLabel2">
                            저항영역을 수정하세요
                          </h2>
                          <button
                            type="button"
                            class="btn-close"
                            data-bs-dismiss="modal"
                            aria-label="Close"
                          ></button>
                        </div>
                        <div class="modal-body">
                          <div
                            class="row mb-3"
                            v-for="(row, idx) in temp_area_points"
                            :key="`${row}_${idx}`"
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
                    class="modal fade"
                    id="exampleModal"
                    tabindex="-1"
                    aria-labelledby="exampleModalLabel"
                    aria-hidden="true"
                  >
                    <div class="modal-dialog">
                      <div class="modal-content">
                        <div class="modal-header">
                          <h2 class="modal-title" id="exampleModalLabel">
                            저항값을 입력하세요
                          </h2>
                          <button
                            type="button"
                            class="btn-close"
                            aria-label="Close"
                          ></button>
                        </div>
                        <div class="modal-body">
                          <div
                            class="row mb-3"
                            v-for="(row, idx) in circuit"
                            :key="`${row}_${idx}`"
                          >
                            <label
                              :for="row['name']"
                              class="col-sm-3 col-form-label"
                              >{{ row["name"] }}</label
                            >
                            <div class="col">
                              <input
                                type="number"
                                class="form-control"
                                :placeholder="row['value']"
                                :id="row['name']"
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
import ImageModify from "../components/ImageModify.vue";
import axios from "axios";

export default {
  component: { CardBody, ImageModify },
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

    let jsonObj = this.$route.query;
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
        let img_data = response.data["img"];
        this.uploaded_img = "data:image/jpg;base64," + img_data;
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
        let img_data = response.data["circuit"];
        localStorage.circuit_img = img_data;
        this.circuit_img = img_data;
      });
    }

    axios({
      url: "http://localhost:3000/resistor",
      method: "get",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((response) => {
      const response_data = response.data.data;
      let temp_arr = [];
      for (let i = 0; i < response_data.length; i++) {
        for (let j = 0; j < response_data[i].length; j++) {
          temp_arr.push(response_data[i][j]);
        }
      }
      this.circuit = temp_arr;
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
      var image = this.$refs["image"].files[0];
      const url = URL.createObjectURL(image);
      this.image = url;
    },
    setResistorArea() {
      if (Object.keys(this.temp_area_points).length === 0) return;

      Object.keys(this.temp_area_points).map((key) => {
        let row = this.temp_area_points[key];
        this.area_points[key] = row;
      });

      let temp = JSON.parse(localStorage.detected_components);
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
          let img_data = "data:image/png;base64," + response.data["circuit"];
          localStorage.circuit_img = response.data["circuit"];
          this.circuit_img = img_data;
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
        if (row["name"] === name) {
          row["value"] = parseInt(event.target.value);
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
          let img_data = response.data["circuit"];

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

          localStorage.circuit_img = img_data;
          this.circuit_img = "data:image/png;base64," + img_data;
          // window.location.reload();
        });
      });
    },
    onImageLoad() {
      console.log("onImageLoad");
      let img = new Image();
      img.src = this.uploaded_img;
      this.scale = localStorage.scale;

      img.onload = () => {
        let width_size = parseInt(img.width * this.scale);
        let height_size = parseInt(img.height * this.scale);
        let resistor_area = this.detected_components["resistor_body"];

        if (Object.keys(resistor_area).length === 0) return;

        this.img_tag.width = width_size + 2;
        this.img_tag.height = height_size + 2;
        this.canvas.width = width_size;
        this.canvas.height = height_size;

        let components = JSON.parse(localStorage.components);
        // Resistor
        Object.keys(components["Resistor"]).forEach((key) => {
          let row = components["Resistor"][key][0];
          let start_coord = row.start_coord;
          let end_coord = row.end_coord;
          console.log(start_coord[0] * this.scale, start_coord[1] * this.scale);
          this.context.beginPath();
          this.context.fillStyle = "red";
          this.context.font = "20px Arial";
          this.context.fillText(
            row.start,
            start_coord[0] * this.scale,
            start_coord[1] * this.scale - 20
          );
          this.context.arc(
            start_coord[0] * this.scale,
            start_coord[1] * this.scale,
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
            end_coord[0] * this.scale,
            end_coord[1] * this.scale,
            5,
            0,
            Math.PI * 2,
            true
          );
          this.context.font = "20px Arial";
          this.context.fillStyle = "orange";
          this.context.fillText(
            row.end,
            end_coord[0] * this.scale,
            end_coord[1] * this.scale + 20
          );
          this.context.fill();
          this.context.closePath();
        });

        Object.keys(components["Line"]).forEach((key) => {
          console.log(key);
        });

        Object.keys(resistor_area).forEach((key) => {
          let row = resistor_area[key];

          let [xmin, ymin, xmax, ymax] = [
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
        let row = points[key];
        let xmin = row.xmin * this.scale;
        let ymin = row.ymin * this.scale;
        let xmax = row.xmax * this.scale;
        let ymax = row.ymax * this.scale;

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

      let currentX = event.offsetX;
      let currentY = event.offsetY;

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

      let x_min = 0;
      let y_min = 0;
      let x_max = 0;
      let y_max = 0;
      let wid = 0;
      let hei = 0;

      x_min = this.startX;
      y_min = this.startY;
      x_max = this.startX + this.finalWidth;
      y_max = this.startY + this.finalHeight;

      let y_dif = y_max - y_min;

      if (y_dif < 0) {
        let [t1, t2] = [x_min, y_min];
        x_min = x_max;
        y_min = y_max;
        x_max = t1;
        y_max = t2;
      }

      x_min /= this.scale;
      y_min /= this.scale;
      x_max /= this.scale;
      y_max /= this.scale;

      wid = Math.abs(this.finalWidth) / this.scale;
      hei = Math.abs(this.finalHeight) / this.scale;

      if (
        (0 <= this.finalWidth && this.finalWidth <= 30) ||
        (0 <= this.finalHeight && this.finalHeight <= 30)
      ) {
        console.log("No save");
        this.isDraging = false;
        this.isDrawing = false;
        this.startX = null;
        this.startY = null;
        this.finalHeight = null;
        this.finalWidth = null;
      } else {
        let new_obj = {
          xmin: x_min,
          ymin: y_min,
          length: hei,
          width: wid,
          xmax: x_max,
          ymax: y_max,
          confidence: 1.0,
          cbRGB: this.cbRGB,
        };

        this.temp_area_points[lastKey] = new_obj;

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
          let row = this.area_points[key];
          let xmin = row.xmin * this.scale;
          let ymin = row.ymin * this.scale;
          let xmax = row.xmax * this.scale;
          let ymax = row.ymax * this.scale;

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
  components: { CardBody },
};
</script>
