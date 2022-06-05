<template>
  <div class="container">
    <h1>Upload</h1>
    <div class="row">
      <div class="col-md-7">
        <ImageModify
          :img_src="image"
          :isSuccess="false"
          @sendData="receiveData"
          @pointCount="checkPointCount"
        ></ImageModify>
      </div>
      <div class="col-md-5" style="margin-bottom: 100px">
        <form
          action="http://localhost:3000/image"
          method="POST"
          enctype="multipart/form-data"
        >
          <div class="mb-3">
            <label for="a" class="form-label">전압 입력</label>
            <input
              type="number"
              class="form-control"
              placeholder="V 단위로 입력하세요"
              @input="setVoltage"
            />
          </div>
          <div class="mb-3">
            <label for="formFile" class="form-label">회로 이미지 입력</label>
            <input
              class="form-control"
              ref="image"
              @change="uploadImg"
              name="file"
              type="file"
              id="file"
              accept="image/*"
            />
          </div>
          <div class="d-grid gap-2">
            <button
              type="button"
              class="btn btn-outline-primary"
              @click.self.prevent="checkState"
            >
              전송
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
<script>
import ImageModify from "../components/ImageModify.vue";
import axios from "axios";

export default {
  components: { ImageModify },
  data() {
    return {
      image: "",
      points: "",
      scale: "",
      image_data: "",
      image_raw: "",
      pointCount: "",
      voltage: null,
      isSuccess: false,
    };
  },
  watch: {},
  methods: {
    setVoltage(event) {
      this.voltage = event.target.value;
    },
    uploadImg() {
      var image = this.$refs["image"].files[0];
      const url = URL.createObjectURL(image);
      this.image_raw = image;
      this.image = url;
    },
    receiveData(data) {
      this.image_data = data;
      this.scale = 0.25;
    },
    checkPointCount(p) {
      this.pointCount = p;
    },
    checkState() {
      if (this.pointCount < 4) {
        console.log(this.pointCount);
      } else {
        if (this.image_data && this.image && this.voltage) {
          this.image_data["img_name"] = this.image_raw.name;
          this.image_data["voltage"] = this.voltage;
          let points = JSON.stringify(this.image_data);

          console.log(points);

          // file은 json에 포함될 수 없음
          let data = new FormData();
          data.append(
            "image",
            new Blob([this.image_raw], { type: "image/jpeg" })
          );
          data.append("data", new Blob([points], { type: "application/json" }));

          axios({
            method: "post",
            url: "/image",
            headers: {
              "Content-Type": "multipart/form-data",
            },
            data: data,
          })
            .then((response) => {
              console.log(response.data);
              this.isSuccess = true;
              localStorage.img = response.data.result_image;
              localStorage.origin_img = response.data.origin_img;
              localStorage.circuit_img = response.data.circuit;

              console.log(response.data.area_points);

              Object.keys(response.data.area_points).map((key) => {
                let row = response.data.area_points[key];
                row.cbRGB = [
                  parseInt(Math.random() * 255),
                  parseInt(Math.random() * 255),
                  parseInt(Math.random() * 255),
                ];
              });

              localStorage.area_points = JSON.stringify(
                response.data.area_points
              );
              localStorage.circuit_analysis = JSON.stringify(
                response.data.circuit_analysis
              );
              localStorage.detected_components = JSON.stringify(
                response.data.detected_components
              );

              localStorage.scale = response.data.scale;

              this.$router.push({
                name: "Result",
                query: response.data,
              });
            })
            .catch((error) => console.log(error));
          this.isSuccess = false;
        } else {
          console.log("empty data");
        }
      }
    },
  },
};
</script>
