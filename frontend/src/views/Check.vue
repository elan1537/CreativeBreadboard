<template>
  <div class="container">
    <div class="row">
      <h1>Check View</h1>
    </div>
    <div class="row">
      <div class="col-1" />
      <div class="col-6">
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
            @mousemove="onMouseMove"
            @mousedown="onMouseDown"
            @mouseup="onMouseUp"
            @keydown="onKeydown"
          />
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
                <th scope="col" />
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="value in detected_line_components"
                :key="value['name']"
              >
                <td scope="col" />
                <td>{{ value['name'] }}</td>

                <td v-if="click_table[value['name']]">
                  <select class="form-select" aria-label="클래스">
                    <option selected value="1">Line</option>
                    <option value="2">Resistor</option>
                  </select>
                </td>
                <td v-else>
                  {{ value['class'] }}
                </td>

                <td v-if="click_table[value['name']]">
                  <input
                    v-model="detected_line_components[value['name']]['start']"
                    type="text"
                    class="form-control"
                    placeholder="시작점"
                    aria-label="시작점"
                  />
                </td>
                <td v-else>
                  {{ value['start'] }}
                </td>

                <td v-if="click_table[value['name']]">
                  <input
                    v-model="detected_line_components[value['name']]['end']"
                    type="text"
                    class="form-control"
                    placeholder="끝점"
                    aria-label="끝점"
                  />
                </td>
                <td v-else>
                  {{ value['end'] }}
                </td>

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
                <td scope="col" />
                <td>{{ value['name'] }}</td>

                <td v-if="click_table[value['name']]">
                  <select class="form-select" aria-label="클래스">
                    <option value="1">Line</option>
                    <option selected value="2">Resistor</option>
                  </select>
                </td>
                <td v-else>
                  {{ value['class'] }}
                </td>

                <td v-if="click_table[value['name']]">
                  <input
                    v-model="
                      detected_resistor_components[value['name']]['start']
                    "
                    type="text"
                    class="form-control"
                    placeholder="시작점"
                    aria-label="시작점"
                  />
                </td>
                <td v-else>
                  {{ value['start'] }}
                </td>

                <td v-if="click_table[value['name']]">
                  <input
                    v-model="detected_resistor_components[value['name']]['end']"
                    type="text"
                    class="form-control"
                    placeholder="끝점"
                    aria-label="끝점"
                  />
                </td>
                <td v-else>
                  {{ value['end'] }}
                </td>

                <td v-if="click_table[value['name']]">
                  <div @click="edit_row(value['name'])">update</div>
                  <div @click="click_table[value['name']] = false">cancel</div>
                </td>
                <td v-else @click="edit_row(value['name'])">edit</td>
              </tr>
              <tr v-if="isAddArea">
                <td scope="col" />
                <td>
                  <input
                    v-model="new_area['name']"
                    type="text"
                    class="form-control"
                    placeholder="컴포넌트 이름"
                    aria-label="컴포넌트 이름"
                  />
                </td>
                <td>
                  <select
                    v-model="new_area['class']"
                    class="form-select"
                    aria-label="클래스"
                  >
                    <option selected value="1">Line</option>
                    <option value="2">Resistor</option>
                  </select>
                </td>
                <td>
                  <input
                    v-model="new_area['start']"
                    type="text"
                    class="form-control"
                    placeholder="시작점"
                    aria-label="시작점"
                  />
                </td>
                <td>
                  <input
                    v-model="new_area['end']"
                    type="text"
                    class="form-control"
                    placeholder="끝점"
                    aria-label="끝점"
                  />
                </td>
                <td>
                  <div @click="checkArea">Add</div>
                  <div @click="cancelArea">Cancel</div>
                </td>
              </tr>
            </tbody>
          </table>
          <button type="button" class="btn btn-primary" @click="addArea">
            영역 추가
          </button>
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
<script>
import ImageModify from '../components/ImageModify.vue';
import axios from 'axios';

export default {
  name: 'CheckView',
  component: { ImageModify },
  data() {
    return {
      uploaded_img: null,
      scale: null,
      detected_line_components: null,
      detected_resistor_components: null,
      detected_unknown_components: null,
      click_table: null,
      isAddArea: false,
      isClick: false,
      prevPoint: [],
      new_area: {
        name: '',
        class: '',
        start: '',
        end: '',
        areaStart: '',
        start_coord: '',
        areaEnd: '',
        end_coord: '',
      },
    };
  },
  created() {
    this.uploaded_img = 'data:image/jpg;base64,' + localStorage.transformedImg;
    this.scale = localStorage.scale;
  },
  mounted() {
    this.img_tag = this.$refs.imageLayer;
    this.canvas = this.$refs.canvas;
    this.context = this.canvas.getContext('2d');

    window.addEventListener(
      'keydown',
      event => {
        event.preventDefault();
        if (event.key === 'Escape') {
          this.context.clearRect(0, 0, 4000, 3000);
        }
      },
      false,
    );
  },
  methods: {
    async areaCheck() {
      const data = localStorage.components;

      await axios({
        url: 'http://localhost:3000/network',
        method: 'post',
        data,
        headers: { 'Content-Type': 'application/json' },
      }).then(() => {
        axios({
          url: 'http://localhost:3000/calc',
          method: 'get',
          headers: { 'Content-Type': 'application/json' },
        }).then(response => {
          localStorage.circuitAnalysis = JSON.stringify(
            response.data.circuit_analysis,
          );
        });
        axios({
          url: 'http://localhost:3000/draw',
          method: 'get',
          headers: { 'Content-Type': 'multipart/form-data' },
        }).then(response => {
          const circuitDiagram =
            'data:image/png;base64,' + response.data.circuit;
          localStorage.circuitDiagram = circuitDiagram;

          this.$router.push({
            name: 'Result',
          });
        });
      });
    },
    async checkArea() {
      const components = JSON.parse(localStorage.components);

      switch (this.new_area.class) {
        case '1':
          console.log('Line');
          this.new_area.class = 'Line';

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: this.new_area.start,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            this.new_area.start_coord = coord;
          });

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: this.new_area.end,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            this.new_area.end_coord = coord;
          });

          this.detected_line_components[this.new_area.name] = this.new_area;
          components.Line = this.detected_line_components;
          break;

        case '2':
          console.log('Resistor');
          this.new_area.class = 'Resistor';
          this.detected_resistor_components[this.new_area.name] = this.new_area;
          components.Resistor = this.detected_resistor_components;
          break;
      }

      this.clearRect(0, 0, 4000, 3000);
      this.set_drawable_area(this.detected_line_components, 'red');
      this.set_drawable_area(this.detected_resistor_components, 'blue');

      localStorage.components = JSON.stringify(components);

      this.isAddArea = false;
    },
    cancelArea() {
      this.context.clearRect(0, 0, 4000, 3000);
      this.set_drawable_area(this.detected_line_components, 'red');
      this.set_drawable_area(this.detected_resistor_components, 'blue');
      this.set_drawable_area(this.detected_unknown_components, 'green');
      this.isAddArea = false;
    },

    async edit_row(id) {
      if (this.click_table[id]) {
        console.log('update!');

        const components = JSON.parse(localStorage.components);
        let updatedComponent;

        if (id.search('L') !== -1) {
          updatedComponent = this.detected_line_components[id];

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: updatedComponent.start,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            console.log(coord);
            this.detected_line_components[id].start_coord = coord;
          });

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: updatedComponent.end,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            this.detected_line_components[id].end_coord = coord;
          });

          console.log(this.detected_line_components[id]);

          components.Line = this.detected_line_components;
          console.log(components.Line[id]);
        } else if (id.search('R') !== -1) {
          updatedComponent = this.detected_resistor_components[id];

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: updatedComponent.start,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            this.detected_resistor_components[id].start_coord = coord;
          });

          await axios({
            url: 'http://localhost:3000/pinmap',
            method: 'get',
            params: {
              pin: updatedComponent.end,
            },
            headers: { 'Content-Type': 'application/json' },
          }).then(response => {
            const coord = response.data.coord;
            this.detected_resistor_components[id].end_coord = coord;
          });

          components.Resistor = this.detected_resistor_components;
          console.log(components.Resistor[id]);
        }

        localStorage.components = JSON.stringify(components);

        const lineComponents = components.Line;
        const resistorComponents = components.Resistor;
        const unknownComponents = components.Unknown;

        this.context.clearRect(0, 0, 3000, 4000);
        this.set_drawable_area(lineComponents, 'red');
        this.set_drawable_area(resistorComponents, 'blue');
        this.set_drawable_area(unknownComponents, 'green');
      } else {
        console.log('edit!');
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
      Object.keys(component).forEach(key => {
        const row = component[key];

        const startCoord = [
          row.start_coord[0] * this.scale,
          row.start_coord[1] * this.scale,
        ];
        const startPin = row.start;
        const endCoord = [
          row.end_coord[0] * this.scale,
          row.end_coord[1] * this.scale,
        ];
        const endPin = row.end;

        const areaStart = [
          row.areaStart[0] * this.scale,
          row.areaStart[1] * this.scale,
        ];
        const areaEnd = [
          row.areaEnd[0] * this.scale,
          row.areaEnd[1] * this.scale,
        ];

        this.draw_arc_component(this.context, 'red', [
          startCoord[0],
          startCoord[1],
        ]);
        this.draw_arc_component(this.context, 'green', [
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
          [areaEnd[0], areaEnd[1]],
        );

        this.draw_text_component(
          this.context,
          'yellow',
          '20px Arial',
          startPin,
          [startCoord[0] - 20, startCoord[1] - 20],
        );
        this.draw_text_component(this.context, 'yellow', '20px Arial', endPin, [
          endCoord[0],
          endCoord[1] - 20,
        ]);
      });
    },
    onImageLoad() {
      const img = new Image();
      img.src = this.uploaded_img;

      img.onload = () => {
        console.log(img.width, img.height);
        const widthSize = parseInt(img.width * this.scale);
        const heightSize = parseInt(img.height * this.scale);

        this.img_tag.width = widthSize + 2;
        this.img_tag.height = heightSize + 2;
        this.canvas.width = widthSize;
        this.canvas.height = heightSize;

        const components = JSON.parse(localStorage.components);

        const lineComponents = components.Line;
        const resistorComponents = components.Resistor;
        const unknownComponents = components.Unknown;

        this.detected_line_components = lineComponents;
        this.detected_resistor_components = resistorComponents;
        this.unknown_components = unknownComponents;

        this.click_table = {};

        Object.keys(lineComponents).forEach(id => {
          this.click_table[id] = false;
        });

        Object.keys(resistorComponents).forEach(id => {
          this.click_table[id] = false;
        });

        Object.keys(unknownComponents).forEach(id => {
          this.click_table[id] = false;
        });

        this.set_drawable_area(lineComponents, 'red');
        this.set_drawable_area(resistorComponents, 'blue');
        this.set_drawable_area(unknownComponents, 'green');
      };
    },
    onMouseDown(event) {
      if (this.isAddArea === true) {
        this.isClicking = true;
        this.prevPoint = [event.offsetX, event.offsetY];
        this.new_area.areaStart = [
          this.prevPoint[0] / 0.25,
          this.prevPoint[1] / 0.25,
        ];

        const [x, y] = [event.offsetX, event.offsetY];
        this.new_area.areaEnd = [x / 0.25, y / 0.25];

        this.draw_rect_component(this.context, 'black', this.prevPoint, [x, y]);
      }
    },
    onMouseUp(event) {
      if (this.isAddArea === true) {
        this.isClicking = false;
      }
    },
    onMouseMove(event) {
      if (this.isClicking === true && this.isAddArea === true) {
        const [x, y] = [event.offsetX, event.offsetY];
        this.new_area.areaEnd = [x / 0.25, y / 0.25];
        this.context.clearRect(0, 0, 4000, 3000);
        this.draw_rect_component(this.context, 'black', this.prevPoint, [x, y]);
      }
    },
    addArea() {
      this.context.clearRect(0, 0, 4000, 3000);
      this.isAddArea = true;
    },
  },
};
</script>
<style>
.row > button {
  margin-top: 30px;
  margin-bottom: 10px;
}
</style>
