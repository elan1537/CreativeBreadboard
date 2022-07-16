<template>
  <div class="container px-4 px-lg-5">
    <!-- Heading Row-->
    <div class="row gx-4 gx-lg-5 align-items-center my-5">
      <div class="col-lg-7">
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
          ></canvas>
        </div>
      </div>
      <div class="col-lg-5">
        <div class="row">
          <img
            class="img-fluid rounded mb-4 mb-lg-4"
            :src="circuit"
            width="600"
            height="800"
            alt="..."
          />
        </div>
        <div class="row mb-3">
          <div class="col">
            <div class="dropdown">
              <div class="d-grid gap-2">
                <button
                  class="btn btn-secondary dropdown-toggle"
                  href="#"
                  role="button"
                  id="dropdownMenuLink"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  전압
                </button>
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuLink">
                  <li
                    v-for="(row, idx) in JSON.parse(
                      circuit_analysis['node_voltage']
                    )"
                    :key="`${row}_${idx}`"
                  >
                    <a class="dropdown-item"
                      >Node-{{ idx }}::{{ parseFloat(row[0]).toFixed(3) }}V</a
                    >
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div class="col">
            <div class="dropdown">
              <div class="d-grid gap-2">
                <button
                  class="btn btn-secondary dropdown-toggle"
                  href="#"
                  role="button"
                  id="dropdownMenuLink"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  전류
                </button>
                <ul class="dropdown-menu" aria-labelledby="dropdownMenuLink">
                  <li>
                    {{
                      parseFloat(circuit_analysis["node_current"]).toFixed(3)
                    }}A
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div class="row">합성저항 :: {{ circuit_analysis["r_th"] }}</div>
      </div>
    </div>
  </div>
</template>
<style></style>
<script>
export default {
  name: "Result",
  data() {
    return {
      img: "data:image/png;base64,",
      circuit: "",
      circuit_analysis: null,
      detected_components: null,
      img_tag: null,
      context: null,
      canvas: null,
      uploaded_img: null,
      scale: null,
    };
  },
  created() {
    if (localStorage.transformedImg) {
      this.img += localStorage.transformedImg;
      this.circuit = localStorage.circuitDiagram;
      this.circuit_analysis = JSON.parse(localStorage.circuitAnalysis);
      this.uploaded_img = this.img;
      this.detected_components = JSON.parse(localStorage.components);
      this.scale = localStorage.scale;
    }
  },
  mounted() {
    this.img_tag = this.$refs.imageLayer;
    this.canvas = this.$refs.canvas;
    this.context = this.canvas.getContext("2d");
  },
  methods: {
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
  },
};
</script>
