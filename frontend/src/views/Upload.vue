<template>
  <div class="container">
    <h1>Upload</h1>
    <div class="row">
      <div class="col-md-7">
        <ImageModify
          :img-src="imageSrc"
          :is-success="false"
          @send-data="receiveData"
          @point-count="checkPointCount"
        />
      </div>
      <div class="col-md-5" style="margin-bottom: 100px">
        <form>
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
              ref="circuitImg"
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
import ImageModify from '../components/ImageModify.vue';
import axios from 'axios';
// import ff from "../../../IMG_5633.txt";

export default {
  name: 'UploadView',
  components: { ImageModify },
  data() {
    return {
      points: '',
      scale: '',
      imageData: {},
      imageSrc: '',
      circuitImage: '',
      pointCount: 0,
      voltage: null,
      isSuccess: false,
    };
  },
  watch: {},
  methods: {
    sendTestdata() {
      axios({
        method: 'post',
        url: 'http://localhost:3000/image',
        headers: {
          'Content-Type': 'application/json',
        },
        data: JSON.stringify({
          isTestDataClick: 'True',
        }),
      }).then(response => {
        this.isSuccess = true;

        localStorage.transformedImg = response.data.transformedImg;
        localStorage.components = JSON.stringify(response.data.components);
        localStorage.scale = response.data.scale;
        localStorage.voltage = response.data.voltage;
        localStorage.basePoint = JSON.stringify(response.data.basePoint);

        this.$router.push({
          name: 'Check',
        });
      });
    },
    setVoltage(event) {
      this.voltage = event.target.value;
    },
    uploadImg() {
      const image = this.$refs.circuitImg.files[0];
      this.imageData.img_name = image.name;
      this.imageData.scale = 0.25;
      this.circuitImage = new Blob([image], { type: 'image/jpeg' });
      this.imageSrc = URL.createObjectURL(image);
    },
    receiveData(data) {
      this.imageData = data;
    },
    checkPointCount(p) {
      this.pointCount = p;
    },
    afterResponse(response) {
      this.isSuccess = true;

      localStorage.transformedImg = response.data.transformedImg;
      localStorage.components = JSON.stringify(response.data.components);
      localStorage.scale = response.data.scale;
      localStorage.voltage = response.data.voltage;
      localStorage.basePoint = JSON.stringify(response.data.basePoint);

      this.$router.push({
        name: 'Check',
      });
    },
    checkState() {
      if (this.pointCount < 4) {
        console.log(this.pointCount);
      } else {
        if (this.imageData && this.voltage) {
          this.imageData.voltage = this.voltage;
          const points = JSON.stringify(this.imageData);
          const formData = new FormData();
          console.log(this.imageData);
          formData.append('points', points);
          formData.append('circuitImage', this.circuitImage);

          axios
            .post('http://localhost:3000/image', formData, {
              headers: { 'Content-Type': 'multipart/form-data' },
            })
            .then(res => this.afterResponse(res))
            .catch(error => {
              console.log(error.toJSON());
            });

          this.isSuccess = false;
        } else {
          console.log('empty data');
        }
      }
    },
  },
};
</script>
