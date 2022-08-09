<template>
  <div class="container">
    <h1>Upload</h1>
    <div class="row">
      <div class="col-md-7">
        <ImageModify
          :img_src="image"
          :is-success="false"
          @send-data="receiveData"
          @point-count="checkPointCount"
        />
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
              id="file"
              ref="image"
              class="form-control"
              name="file"
              type="file"
              accept="image/*"
              @change="uploadImg"
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
    <div class="row">
      <div class="mb-3">
        <button
          type="button"
          class="btn btn-outline-primary"
          @click="sendTestdata"
        >
          테스트 데이터
        </button>
      </div>
    </div>
  </div>
</template>
<script>
import ImageModify from "../components/ImageModify.vue";
import axios from "axios";
// import ff from "../../../IMG_5633.txt";

export default {
  name: "UploadView",
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
    sendTestdata() {
      axios({
        method: "post",
        url: "http://localhost:3000/image",
        headers: {
          "Content-Type": "application/json",
        },
        data: JSON.stringify({
          isTestDataClick: "True",
        }),
      }).then((response) => {
        this.isSuccess = true;

        localStorage.transformedImg = response.data.transformedImg;
        localStorage.components = JSON.stringify(response.data.components);
        localStorage.scale = response.data.scale;
        localStorage.voltage = response.data.voltage;
        localStorage.basePoint = JSON.stringify(response.data.basePoint);

        this.$router.push({
          name: "Check",
        });
      });
    },
    setVoltage(event) {
      this.voltage = event.target.value;
    },
    uploadImg() {
      const image = this.$refs.image.files[0];
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
          this.image_data.img_name = this.image_raw.name;
          this.image_data.voltage = this.voltage;
          const points = JSON.stringify(this.image_data);
          console.log(points);

          console.log(this.image_raw);

          // file은 json에 포함될 수 없음
          const data = new FormData();
          data.append(
            "image",
            new Blob([this.image_raw], { type: "image/jpeg" })
          );
          data.append("data", new Blob([points], { type: "application/json" }));

          axios({
            method: "post",
            url: "http://localhost:3000/image",
            headers: {
              "Content-Type": "multipart/form-data",
            },
            data,
          })
            .then((response) => {
              this.isSuccess = true;

              localStorage.transformedImg = response.data.transformedImg;
              localStorage.components = JSON.stringify(
                response.data.components
              );
              localStorage.scale = response.data.scale;
              localStorage.voltage = response.data.voltage;
              localStorage.basePoint = JSON.stringify(response.data.basePoint);

              this.$router.push({
                name: "Check",
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
