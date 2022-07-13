<template>
  <div class="container">
    <h1>Check View</h1>
    <div class="row">
      <div class="col ms-auto">
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
    </div>
  </div>
</template>
<script>
import ImageModify from "../components/ImageModify.vue";
export default {
  component: { ImageModify },
  data() {
    return {
      uploaded_img: null,
      scale: null,
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

        const set_drawable_area = (component) => {
          Object.keys(component).forEach((key) => {
            let row = component[key];
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

            console.log(startCoord, endCoord);

            const draw_text_component = (context, color, font, text, coord) => {
              context.beginPath();
              context.fillStyle = color;
              context.font = font;
              context.fillText(text, coord[0], coord[1]);
              context.fill();
              context.closePath();
            };

            const draw_arc_component = (context, color, coord) => {
              context.beginPath();
              context.fillStyle = color;
              context.arc(coord[0], coord[1], 5, 0, Math.PI * 2, true);
              context.fill();
              context.closePath();
            };

            const draw_rect_component = (context, color, start, end) => {
              context.beginPath();
              context.strokeStyle = color;
              context.lineWidth = 3;
              context.rect(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1]
              );
              context.stroke();
              context.closePath();
            };

            draw_text_component(this.context, "blue", "20px Arial", startPin, [
              startCoord[0] - 20,
              startCoord[1] - 20,
            ]);
            draw_text_component(this.context, "blue", "20px Arial", endPin, [
              endCoord[0],
              endCoord[1] - 20,
            ]);

            draw_arc_component(this.context, "red", [
              startCoord[0],
              startCoord[1],
            ]);
            draw_arc_component(this.context, "green", [
              endCoord[0],
              endCoord[1],
            ]);

            draw_rect_component(
              this.context,
              "orange",
              [areaStart[0], areaStart[1]],
              [areaEnd[0], areaEnd[1]]
            );
          });
        };

        set_drawable_area(line_components);
        set_drawable_area(resistor_components);
        set_drawable_area(unknown_components);

        // let resistor_area = this.detected_components["resistor_body"];
        // if (Object.keys(resistor_area).length === 0) return;

        // let components = JSON.parse(localStorage.components);
        // // Resistor
        // Object.keys(components["Resistor"]).forEach((key) => {
        //   let row = components["Resistor"][key][0];
        //   let start_coord = row.start_coord;
        //   let end_coord = row.end_coord;
        //   console.log(start_coord[0] * this.scale, start_coord[1] * this.scale);
        //   this.context.beginPath();
        //   this.context.fillStyle = "blue";
        //   this.context.font = "20px Arial";
        //   this.context.fillText(
        //     row.start_pin,
        //     start_coord[0] * this.scale,
        //     start_coord[1] * this.scale - 20
        //   );
        //   this.context.arc(
        //     start_coord[0] * this.scale,
        //     start_coord[1] * this.scale,
        //     5,
        //     0,
        //     Math.PI * 2,
        //     true
        //   );
        //   this.context.fill();
        //   this.context.closePath();
        //   this.context.beginPath();
        //   this.context.fillStyle = "blue";
        //   this.context.arc(
        //     end_coord[0] * this.scale,
        //     end_coord[1] * this.scale,
        //     5,
        //     0,
        //     Math.PI * 2,
        //     true
        //   );
        //   this.context.font = "20px Arial";
        //   this.context.fillText(
        //     row.end_pin,
        //     end_coord[0] * this.scale,
        //     end_coord[1] * this.scale + 20
        //   );
        //   this.context.fill();
        //   this.context.closePath();
        // });

        // Object.keys(components["Line"]).forEach((key) => {
        //   console.log(key);
        // });

        // Object.keys(resistor_area).forEach((key) => {
        //   let row = resistor_area[key];

        //   let [xmin, ymin, xmax, ymax] = [
        //     row.xmin * this.scale,
        //     row.ymin * this.scale,
        //     row.xmax * this.scale,
        //     row.ymax * this.scale,
        //   ];

        //   this.context.beginPath();
        //   this.context.lineWidth = 4;
        //   this.context.strokeStyle = `rgb(${row.cbRGB[0]}, ${row.cbRGB[1]}, ${row.cbRGB[2]})`;
        //   this.context.rect(xmin, ymin, xmax - xmin, ymax - ymin);
        //   this.context.stroke();
        //   this.context.closePath();
        // });
      };
    },
    onClick(event) {
      console.log(event.offsetX, event.offsetY);
    },
  },
};
</script>
