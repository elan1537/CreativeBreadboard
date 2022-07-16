<template>
  <div class="container">
    <div class="row">
      <h1>Check View</h1>
    </div>
    <div class="row">
      <div class="col-1"></div>
      <div class="col-6">
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
            @click="onClick"
          ></canvas>
        </div>
      </div>
      <div class="col-5">
        <div class="row">
          <table class="table table-hover">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">컴포넌트 이름</th>
                <th scope="col">컴포넌트 클래스</th>
                <th scope="col">연결핀 1</th>
                <th scope="col">연결핀 2</th>
                <th scope="col"></th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="value in detected_line_components"
                :key="value['name']"
              >
                <td scope="col"></td>
                <td>{{ value["name"] }}</td>

                <td v-if="click_table[value['name']]">
                  <select class="form-select" aria-label="클래스">
                    <option selected value="1">Line</option>
                    <option value="2">Resistor</option>
                  </select>
                </td>
                <td v-else>{{ value["class"] }}</td>

                <td v-if="click_table[value['name']]">
                  <input
                    type="text"
                    class="form-control"
                    placeholder="시작점"
                    aria-label="시작점"
                    v-model="detected_line_components[value['name']]['start']"
                  />
                </td>
                <td v-else>{{ value["start"] }}</td>

                <td v-if="click_table[value['name']]">
                  <input
                    type="text"
                    class="form-control"
                    placeholder="끝점"
                    aria-label="끝점"
                    v-model="detected_line_components[value['name']]['end']"
                  />
                </td>
                <td v-else>{{ value["end"] }}</td>

                <td v-if="click_table[value['name']]">
                  <div @click="edit_row(value['name'])">update</div>
                  <div @click="click_table[value['name']] = false">cancel</div>
                </td>

                <td v-else @click="edit_row(value['name'])">edit</td>
              </tr>
              <tr
                v-for="value in detected_resistor_components"
                :key="value['name']"
              >
                <td scope="col"></td>
                <td>{{ value["name"] }}</td>

                <td v-if="click_table[value['name']]">
                  <select class="form-select" aria-label="클래스">
                    <option value="1">Line</option>
                    <option selected value="2">Resistor</option>
                  </select>
                </td>
                <td v-else>{{ value["class"] }}</td>

                <td v-if="click_table[value['name']]">
                  <input
                    type="text"
                    class="form-control"
                    placeholder="시작점"
                    aria-label="시작점"
                    v-model="
                      detected_resistor_components[value['name']]['start']
                    "
                  />
                </td>
                <td v-else>{{ value["start"] }}</td>

                <td v-if="click_table[value['name']]">
                  <input
                    type="text"
                    class="form-control"
                    placeholder="끝점"
                    aria-label="끝점"
                    v-model="detected_resistor_components[value['name']]['end']"
                  />
                </td>
                <td v-else>{{ value["end"] }}</td>

                <td v-if="click_table[value['name']]">
                  <div @click="edit_row(value['name'])">update</div>
                  <div @click="click_table[value['name']] = false">cancel</div>
                </td>
                <td v-else @click="edit_row(value['name'])">edit</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="row">
      <button type="button" class="btn btn-primary" @click="areaCheck">
        영역 확인
      </button>
    </div>
  </div>
</template>
<style>
.row > button {
  margin-top: 30px;
  margin-bottom: 10px;
}
</style>
<script>
import ImageModify from "../components/ImageModify.vue";
import axios from "axios";

export default {
  component: { ImageModify },
  data() {
    return {
      uploaded_img: null,
      scale: null,
      detected_line_components: null,
      detected_resistor_components: null,
      detected_unknown_components: null,
      click_table: null,
    };
  },
  created() {
    this.uploaded_img = "data:image/jpg;base64," + localStorage.transformedImg;
    this.scale = localStorage.scale;
  },
  mounted() {
    this.img_tag = this.$refs.imageLayer;
    this.canvas = this.$refs.canvas;
    this.context = this.canvas.getContext("2d");
  },
  methods: {
    async areaCheck() {
      console.log("Are you sure?");

      let data = localStorage.components;
      console.log(data);

      await axios({
        url: "/network",
        method: "post",
        data: data,
        headers: { "Content-Type": "application/json" },
      }).then(() => {
        axios({
          url: "/calc",
          method: "get",
          headers: { "Content-Type": "application/json" },
        }).then((response) => {
          localStorage.circuitAnalysis = JSON.stringify(
            response.data.circuit_analysis
          );
        });
        axios({
          url: "/draw",
          method: "get",
          headers: { "Content-Type": "multipart/form-data" },
        }).then((response) => {
          let circuitDiagram =
            "data:image/png;base64," + response.data["circuit"];
          localStorage.circuitDiagram = circuitDiagram;

          this.$router.push({
            name: "Result",
          });
        });
      });
    },

    async edit_row(id) {
      if (this.click_table[id]) {
        console.log("update!");

        let components = JSON.parse(localStorage.components);
        let updatedComponent;

        if (id.search("L") !== -1) {
          updatedComponent = this.detected_line_components[id];

          await axios({
            url: "/pinmap",
            method: "get",
            params: {
              pin: updatedComponent.start,
            },
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            let coord = response.data.coord;
            console.log(coord);
            this.detected_line_components[id]["start_coord"] = coord;
          });

          await axios({
            url: "/pinmap",
            method: "get",
            params: {
              pin: updatedComponent.end,
            },
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            let coord = response.data.coord;
            this.detected_line_components[id]["end_coord"] = coord;
          });

          console.log(this.detected_line_components[id]);

          components["Line"] = this.detected_line_components;
          console.log(components["Line"][id]);
        } else if (id.search("R") !== -1) {
          updatedComponent = this.detected_resistor_components[id];

          await axios({
            url: "/pinmap",
            method: "get",
            params: {
              pin: updatedComponent.start,
            },
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            let coord = response.data.coord;
            this.detected_resistor_components[id]["start_coord"] = coord;
          });

          await axios({
            url: "/pinmap",
            method: "get",
            params: {
              pin: updatedComponent.end,
            },
            headers: { "Content-Type": "application/json" },
          }).then((response) => {
            let coord = response.data.coord;
            this.detected_resistor_components[id]["end_coord"] = coord;
          });

          components["Resistor"] = this.detected_resistor_components;
          console.log(components["Resistor"][id]);
        }

        localStorage.components = JSON.stringify(components);

        let line_components = components["Line"];
        let resistor_components = components["Resistor"];
        let unknown_components = components["Unknown"];

        this.context.clearRect(0, 0, 3000, 4000);
        this.set_drawable_area(line_components, "red");
        this.set_drawable_area(resistor_components, "blue");
        this.set_drawable_area(unknown_components, "green");
      } else {
        console.log("edit!");
      }

      this.click_table[id] = !this.click_table[id];
    },
    draw_text_component(context, color, font, text, coord) {
      context.beginPath();
      context.fillStyle = color;
      context.font = font;
      context.fillText(text, coord[0], coord[1]);
      context.fill();
      context.closePath();
    },
    draw_arc_component(context, color, coord) {
      context.beginPath();
      context.fillStyle = color;
      context.arc(coord[0], coord[1], 5, 0, Math.PI * 2, true);
      context.fill();
      context.closePath();
    },
    draw_rect_component(context, color, start, end) {
      context.beginPath();
      context.strokeStyle = color;
      context.lineWidth = 3;
      context.rect(start[0], start[1], end[0] - start[0], end[1] - start[1]);
      context.stroke();
      context.closePath();
    },
    set_drawable_area(component, color) {
      Object.keys(component).forEach((key) => {
        let row = component[key];

        // console.log(row.start_coord);

        let startCoord = [
          row.start_coord[0] * this.scale,
          row.start_coord[1] * this.scale,
        ];
        let startPin = row.start;
        let endCoord = [
          row.end_coord[0] * this.scale,
          row.end_coord[1] * this.scale,
        ];
        let endPin = row.end;

        let areaStart = [
          row.areaStart[0] * this.scale,
          row.areaStart[1] * this.scale,
        ];
        let areaEnd = [
          row.areaEnd[0] * this.scale,
          row.areaEnd[1] * this.scale,
        ];

        this.draw_arc_component(this.context, "red", [
          startCoord[0],
          startCoord[1],
        ]);
        this.draw_arc_component(this.context, "green", [
          endCoord[0],
          endCoord[1],
        ]);

        // if (row.class === "Line") {
        //   let start_endAreaStart = [
        //     row.start_endAreaStart[0] * this.scale,
        //     row.start_endAreaStart[1] * this.scale,
        //   ];
        //   let start_endAreaEnd = [
        //     row.start_endAreaEnd[0] * this.scale,
        //     row.start_endAreaEnd[1] * this.scale,
        //   ];
        //   let end_endAreaStart = [
        //     row.end_endAreaStart[0] * this.scale,
        //     row.end_endAreaStart[1] * this.scale,
        //   ];
        //   let end_endAreaEnd = [
        //     row.end_endAreaEnd[0] * this.scale,
        //     row.end_endAreaEnd[1] * this.scale,
        //   ];
        //   this.draw_rect_component(
        //     this.context,
        //     color,
        //     [start_endAreaStart[0], start_endAreaStart[1]],
        //     [start_endAreaEnd[0], start_endAreaEnd[1]]
        //   );
        //   this.draw_rect_component(
        //     this.context,
        //     color,
        //     [end_endAreaStart[0], end_endAreaStart[1]],
        //     [end_endAreaEnd[0], end_endAreaEnd[1]]
        //   );
        // }

        this.draw_rect_component(
          this.context,
          color,
          [areaStart[0], areaStart[1]],
          [areaEnd[0], areaEnd[1]]
        );

        this.draw_text_component(
          this.context,
          "yellow",
          "20px Arial",
          startPin,
          [startCoord[0] - 20, startCoord[1] - 20]
        );
        this.draw_text_component(this.context, "yellow", "20px Arial", endPin, [
          endCoord[0],
          endCoord[1] - 20,
        ]);
      });
    },
    onImageLoad() {
      let img = new Image();
      img.src = this.uploaded_img;

      img.onload = () => {
        console.log(img.width, img.height);
        let width_size = parseInt(img.width * this.scale);
        let height_size = parseInt(img.height * this.scale);

        this.img_tag.width = width_size + 2;
        this.img_tag.height = height_size + 2;
        this.canvas.width = width_size;
        this.canvas.height = height_size;

        let components = JSON.parse(localStorage.components);

        let line_components = components["Line"];
        let resistor_components = components["Resistor"];
        let unknown_components = components["Unknown"];

        this.detected_line_components = line_components;
        this.detected_resistor_components = resistor_components;
        this.unknown_components = unknown_components;

        this.click_table = {};

        Object.keys(line_components).forEach((id) => {
          this.click_table[id] = false;
        });

        Object.keys(resistor_components).forEach((id) => {
          this.click_table[id] = false;
        });

        Object.keys(unknown_components).forEach((id) => {
          this.click_table[id] = false;
        });

        this.set_drawable_area(line_components, "red");
        this.set_drawable_area(resistor_components, "blue");
        this.set_drawable_area(unknown_components, "green");
      };
    },
    onClick(event) {
      console.log(event.offsetX, event.offsetY);
    },
    onMouseDown(event) {
      console.log(event);
    },
    onMouseMove(event) {
      console.log(event);
    },
  },
};
</script>
