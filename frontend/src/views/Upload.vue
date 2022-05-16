<template>
  <div class="container">
    <h1>Upload</h1>
    <div class="row">
      <div class="col-md-7">
        <ImageModify :img_src="image"></ImageModify>
      </div>
      <div class="col-md-5">
        <form action="/image" method="POST" enctype="multipart/form-data">
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
            <button type="submit" class="btn btn-outline-primary">전송</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
<script>
import ImageModify from "../components/ImageModify.vue";
export default {
  components: { ImageModify },
  data() {
    return {
      image: "",
    };
  },
  methods: {
    uploadImg() {
      var image = this.$refs["image"].files[0];
      const url = URL.createObjectURL(image);
      this.image = url;

      console.log(this.image);
    },
  },
};
</script>
