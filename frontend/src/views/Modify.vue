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
                    @click="setResistorValue"
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
                            data-bs-dismiss="modal"
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
                            aria-label="Close"
                          >
                            Close
                          </button>
                          <button
                            type="button"
                            class="btn btn-primary"
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
      circuit: "",
      uploaded_img: "",
      title_1: "저항영역 수정하기",
      title_2: "저항값 수정하기",
      scale: null,
      canvas: null,
      context: null,
      img_tag: null,
      area_points: null,
      circuit_img: null,
      isDrawing: false,
      isDraging: false,
      startX: null,
      startY: null,
      cbRGB: null,
      finalWidth: null,
      finalHeight: null,
    };
  },
  created() {
    if (localStorage.circuit_img) {
      this.circuit_img = "data:image/png;base64," + localStorage.circuit_img;
    } else {
      axios({
        url: "/draw",
        method: "get",
        headers: { "Content-Type": "multipart/form-data" },
      }).then((response) => {
        let img_data = response.data["circuit"];

        localStorage.circuit_img = img_data;
        this.circuit_img += img_data;
      });
    }

    axios({
      url: "/resistor",
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
    this.uploaded_img = "data:image/png;base64," + localStorage.origin_img;
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
        url: "/resistor",
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

          localStorage.circuit_img = img_data;
          this.circuit_img = "data:image/png;base64," + img_data;
          window.location.reload();
        });
      });
    },
    onImageLoad() {
      let img = new Image();
      img.src = this.uploaded_img;
      this.scale = localStorage.scale;
      this.area_points = JSON.parse(localStorage.area_points);

      img.onload = () => {
        let width_size = parseInt(img.width * this.scale);
        let height_size = parseInt(img.height * this.scale);
        this.img_tag.width = width_size + 2;
        this.img_tag.height = height_size + 2;
        this.canvas.width = width_size;
        this.canvas.height = height_size;

        Object.keys(this.area_points).forEach((key) => {
          let row = this.area_points[key];

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
    onMove(event) {
      if (!this.isDraging) return;

      if (this.isDrawing) {
        let currentX = event.offsetX;
        let currentY = event.offsetY;

        this.finalWidth = currentX - this.startX;
        this.finalHeight = currentY - this.startY;

        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

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
      }
    },
    onDown(event) {
      console.log(event.offsetX, event.offsetY);
      Object.keys(this.area_points).map((key) => {
        let row = this.area_points[key];

        let [xmin, ymin, xmax, ymax] = [
          row.xmin * this.scale,
          row.ymin * this.scale,
          row.xmax * this.scale,
          row.ymax * this.scale,
        ];

        console.log(xmin, ymin, xmax, ymax, event.offsetX, event.offsetY);

        console.log(xmin <= event.offsetX);

        if (
          xmin <= event.offsetX &&
          event.offsetX <= xmax &&
          ymin <= event.offsetY &&
          event.ofsetY <= ymax
        ) {
          console.log(key, "area");
        }
      });

      this.isDraging = true;

      if (!this.isDrawing) {
        this.isDrawing = true;
        this.startX = event.offsetX;
        this.startY = event.offsetY;
        this.cbRGB = [
          parseInt(Math.random() * 255),
          parseInt(Math.random() * 255),
          parseInt(Math.random() * 255),
        ];
      } else {
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
      }
    },
    stopDrag() {
      let lastKey = Object.keys(this.area_points);
      lastKey = parseInt(lastKey[lastKey.length - 1]);
      lastKey += 1;

      // 만약 뺐는데 음수이면 반대..
      // 아니면 양수
      // 그려질 때 거꾸로 그려짐..

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

      this.area_points[lastKey] = new_obj;

      this.isDraging = false;
      this.startX = null;
      this.startY = null;
      this.finalHeight = null;
      this.finalWidth = null;
      this.isDrawing = false;
    },
  },
  components: { CardBody },
};
</script>
